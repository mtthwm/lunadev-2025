use std::{
    net::SocketAddrV4,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Exclusive,
    },
    time::{Duration, Instant},
};

use image::RgbImage;
use lunabot_lib::{
    make_negotiation, ArmAction, ArmParameters, Audio, AutonomyAction, CameraMessage,
    ExecutiveArmAction, ImportantMessage, LunaNegotiation, Odometry, Steering,
};
use networking::{new_client, ConnectionError, NetworkConnector, NetworkNode};
use ordered_float::NotNan;
use serde::Deserialize;
use unros::{
    anyhow,
    logging::dump::{ScalingFilter, VideoDataDump},
    node::{AsyncNode, SyncNode},
    pubsub::{
        subs::Subscription, MonoPublisher, Publisher, PublisherRef, Subscriber, WatchSubscriber,
    },
    runtime::RuntimeContext,
    setup_logging,
    tokio::{self, task::spawn_blocking},
    DontDrop, ShouldNotDrop,
};

use crate::{
    audio::{pause_buzz, pause_music, play_buzz, play_music, MIC_PUB},
    CAMERA_HEIGHT, CAMERA_WIDTH, EMPTY_ROW, MAX_CAMERA_COUNT, ROW_DATA_LENGTH, ROW_LENGTH,
};

#[derive(Deserialize)]
struct TelemetryConfig {
    #[serde(default = "default_server_addr")]
    server_addr: SocketAddrV4,
}

fn default_server_addr() -> SocketAddrV4 {
    let addr = std::env::var("SERVER_ADDR").unwrap_or("192.168.0.100:43721".into());
    addr.parse()
        .expect("SERVER_ADDR must be a valid IP address and port!")
}

/// A remote connection to `Lunabase`
#[derive(ShouldNotDrop)]
pub struct Telemetry {
    network_node: NetworkNode,
    network_connector: NetworkConnector,
    pub server_addr: SocketAddrV4,
    pub camera_delta: Duration,
    steering_signal: Publisher<Steering>,
    arm_signal: Publisher<ArmParameters<ArmAction>>,
    executive_arm_signal: Publisher<ArmParameters<ExecutiveArmAction>>,
    autonomy_signal: Publisher<AutonomyAction>,
    dont_drop: DontDrop<Self>,
    negotiation: LunaNegotiation,
    video_addr: SocketAddrV4,
    cam_width: u32,
    cam_height: u32,
    cam_fps: usize,
    camera_subs: Vec<WatchSubscriber<RgbImage>>,
    odometry_sub: Option<PublisherRef<Odometry>>,
}

impl Telemetry {
    pub async fn new(
        cam_fps: usize,
        camera_subs: Vec<WatchSubscriber<RgbImage>>,
    ) -> anyhow::Result<Self> {
        assert!(camera_subs.len() <= MAX_CAMERA_COUNT, "Too many cameras!");
        let config: TelemetryConfig = unros::get_env()?;
        let mut video_addr = config.server_addr;
        video_addr.set_port(video_addr.port() + 1);

        let (network_node, network_connector) = new_client()?;

        Ok(Self {
            network_node,
            network_connector,
            server_addr: config.server_addr,
            steering_signal: Publisher::default(),
            arm_signal: Publisher::default(),
            executive_arm_signal: Publisher::default(),
            autonomy_signal: Publisher::default(),
            camera_delta: Duration::from_millis((1000 / cam_fps) as u64),
            dont_drop: DontDrop::new("telemetry"),
            negotiation: make_negotiation(),
            cam_width: CAMERA_WIDTH * ROW_LENGTH as u32,
            cam_height: CAMERA_HEIGHT * MAX_CAMERA_COUNT.div_ceil(ROW_LENGTH) as u32,
            video_addr,
            cam_fps,
            camera_subs,
            odometry_sub: None,
        })
    }

    pub fn steering_pub(&self) -> PublisherRef<Steering> {
        self.steering_signal.get_ref()
    }

    pub fn autonomy_pub(&self) -> PublisherRef<AutonomyAction> {
        self.autonomy_signal.get_ref()
    }

    pub fn arm_pub(&self) -> PublisherRef<ArmParameters<ArmAction>> {
        self.arm_signal.get_ref()
    }

    pub fn exec_arm_pub(&self) -> PublisherRef<ArmParameters<ExecutiveArmAction>> {
        self.executive_arm_signal.get_ref()
    }

    pub fn odometry_sub(&mut self, pubref: PublisherRef<Odometry>) {
        self.odometry_sub = Some(pubref);
    }
}

impl AsyncNode for Telemetry {
    type Result = anyhow::Result<()>;

    async fn run(mut self, context: RuntimeContext) -> anyhow::Result<()> {
        setup_logging!(context);
        // self.network_node
        //     .get_intrinsics()
        //     .manually_run(context.get_name().clone());

        self.dont_drop.ignore_drop = true;
        let sdp: Arc<str> =
            Arc::from(VideoDataDump::generate_sdp(self.video_addr).into_boxed_str());
        let enable_camera = Arc::new(AtomicBool::default());
        let enable_camera2 = enable_camera.clone();

        let context2 = context.clone();

        let (swap_sender, swap_receiver) = std::sync::mpsc::channel();
        let mut swap_receiver = Exclusive::new(swap_receiver);

        let cam_fut = async {
            loop {
                let mut video_dump;
                loop {
                    if context2.is_runtime_exiting() {
                        return Ok(());
                    }
                    if enable_camera.load(Ordering::Relaxed) {
                        loop {
                            match VideoDataDump::new_rtp(
                                self.cam_width,
                                self.cam_height,
                                self.cam_width,
                                self.cam_height,
                                ScalingFilter::Neighbor,
                                self.video_addr,
                                self.cam_fps,
                                &context2,
                            ) {
                                Ok(x) => {
                                    video_dump = x;
                                    break;
                                }
                                Err(e) => error!("Failed to create video dump: {e}"),
                            }
                            let start_service = Instant::now();
                            while start_service.elapsed().as_millis() < 2000 {
                                if context2.is_runtime_exiting() {
                                    return Ok(());
                                }
                                tokio::time::sleep(self.camera_delta).await;
                            }
                        }
                        break;
                    }
                    tokio::time::sleep(self.camera_delta).await;
                }
                let mut start_service = Instant::now();
                loop {
                    if context2.is_runtime_exiting() {
                        return Ok(());
                    }
                    if !enable_camera.load(Ordering::Relaxed) {
                        drop(video_dump);
                        break;
                    }
                    while let Ok((first, second)) = swap_receiver.get_mut().try_recv() {
                        if first < self.camera_subs.len() && second < self.camera_subs.len() {
                            self.camera_subs.swap(first, second);
                        }
                    }
                    let mut updated = false;
                    self.camera_subs
                        .iter_mut()
                        .for_each(|sub| updated |= WatchSubscriber::try_update(sub));
                    if updated {
                        for row in self.camera_subs.chunks(ROW_LENGTH) {
                            for y in 0..CAMERA_HEIGHT as usize {
                                for i in 0..ROW_LENGTH {
                                    let row_data = if let Some(img) = row.get(i) {
                                        img.deref()
                                            .split_at(ROW_DATA_LENGTH * y)
                                            .1
                                            .split_at(ROW_DATA_LENGTH)
                                            .0
                                    } else {
                                        &EMPTY_ROW
                                    };
                                    if let Err(e) = video_dump.write_raw(row_data).await {
                                        error!("Failed to write camera data: {e}");
                                    }
                                }
                            }
                        }
                        // println!("{}", MAX_CAMERA_COUNT
                        // .next_multiple_of(ROW_LENGTH)
                        // .saturating_sub(self.camera_subs.len()));
                        for _ in 0..MAX_CAMERA_COUNT
                            .next_multiple_of(ROW_LENGTH)
                            .saturating_sub(self.camera_subs.len().next_multiple_of(ROW_LENGTH))
                        {
                            for _ in 0..CAMERA_HEIGHT as usize {
                                if let Err(e) = video_dump.write_raw(&EMPTY_ROW).await {
                                    error!("Failed to write camera data: {e}");
                                }
                            }
                        }
                    }

                    let elapsed = start_service.elapsed();
                    start_service += elapsed;
                    tokio::time::sleep(self.camera_delta).await;
                }
            }
        };
        let enable_camera = enable_camera2;

        let peer_fut = async {
            loop {
                info!("Connecting to lunabase...");
                let peer = loop {
                    match self
                        .network_connector
                        .connect_to(self.server_addr.into(), &12u8)
                        .await
                    {
                        Ok(x) => break x,
                        Err(ConnectionError::ServerDropped) => return Ok(()),
                        Err(ConnectionError::Timeout) => {}
                    };
                };
                let (important, camera, odometry, controls, audio, audio_controls) =
                    match peer.negotiate(&self.negotiation).await {
                        Ok(x) => x,
                        Err(e) => {
                            error!("Failed to negotiate with lunabase!: {e:?}");
                            continue;
                        }
                    };
                enable_camera.store(true, Ordering::Relaxed);
                info!("Connected to lunabase!");

                if let Some(mic_pub) = MIC_PUB.get() {
                    mic_pub.accept_subscription(audio.create_unreliable_subscription());
                }

                if let Some(odometry_sub) = self.odometry_sub.clone() {
                    let mut i = 0usize;
                    odometry_sub.accept_subscription(
                        odometry
                            .create_unreliable_subscription()
                            .filter_map(move |x| {
                                i = (i + 1) % 6;
                                if i == 1 {
                                    Some(x)
                                } else {
                                    None
                                }
                            }),
                    );
                }

                let important_fut = async {
                    let mut _important_pub =
                        MonoPublisher::from(important.create_reliable_subscription());
                    let important_sub = Subscriber::new(8);
                    important.accept_subscription(important_sub.create_subscription());

                    loop {
                        let Some(result) = important_sub.recv_or_closed().await else {
                            break;
                        };
                        let msg = match result {
                            Ok(x) => x,
                            Err(e) => {
                                error!("Error receiving important msg: {e}");
                                continue;
                            }
                        };
                        match msg {
                            ImportantMessage::EnableCamera => {
                                enable_camera.store(true, Ordering::Relaxed)
                            }
                            ImportantMessage::DisableCamera => {
                                enable_camera.store(false, Ordering::Relaxed)
                            }
                            ImportantMessage::Autonomy(action) => {
                                self.autonomy_signal.set(action);
                            }
                            ImportantMessage::ExecutiveArmAction(action) => {
                                self.executive_arm_signal.set(action);
                            }
                        }
                    }
                };

                let steering_fut = async {
                    let mut controls_pub =
                        MonoPublisher::from(controls.create_unreliable_subscription());
                    let controls_sub = Subscriber::new(1);
                    controls.accept_subscription(controls_sub.create_subscription());

                    loop {
                        let Some(result) = controls_sub.recv_or_closed().await else {
                            break;
                        };
                        let controls = match result {
                            Ok(x) => x,
                            Err(e) => {
                                error!("Error receiving steering: {e}");
                                continue;
                            }
                        };
                        controls_pub.set(controls);
                        self.steering_signal.set(Steering::from_drive_and_steering(
                            NotNan::new(controls.drive as f32 / 127.0).unwrap(),
                            NotNan::new(controls.steering as f32 / 127.0).unwrap(),
                        ));
                        self.arm_signal.set(controls.arm_params);
                    }
                };

                let camera_fut = async {
                    let camera_pub = Publisher::default();
                    let camera_sub = Subscriber::new(1);
                    camera.accept_subscription(camera_sub.create_subscription());
                    camera_pub.accept_subscription(camera.create_reliable_subscription());
                    camera_pub.set(CameraMessage::Sdp(sdp.clone()));

                    loop {
                        let Some(result) = camera_sub.recv_or_closed().await else {
                            break;
                        };
                        let msg = match result {
                            Ok(x) => x,
                            Err(e) => {
                                error!("Error receiving camera msg: {e}");
                                continue;
                            }
                        };

                        match msg {
                            CameraMessage::Sdp(_) => {}
                            CameraMessage::Swap(first, second) => {
                                let _ = swap_sender.send((first, second));
                            }
                        }
                    }
                };

                let audio_fut = async {
                    let audio_sub = Subscriber::new(1);
                    audio_controls.accept_subscription(audio_sub.create_subscription());

                    loop {
                        let Some(result) = audio_sub.recv_or_closed().await else {
                            break;
                        };
                        let msg = match result {
                            Ok(x) => x,
                            Err(e) => {
                                error!("Error receiving audio msg: {e}");
                                continue;
                            }
                        };

                        match msg {
                            Audio::PlayBuzz => play_buzz(),
                            Audio::PauseBuzz => pause_buzz(),
                            Audio::PlayMusic => play_music(),
                            Audio::PauseMusic => pause_music(),
                        }
                    }
                };

                tokio::select! {
                    _ = steering_fut => {}
                    _ = camera_fut => {}
                    _ = important_fut => {}
                    _ = audio_fut => {}
                }
                self.steering_signal.set(Steering::default());
                self.arm_signal.set(ArmParameters::default());
                error!("Disconnected from lunabase!");
                enable_camera.store(false, Ordering::Relaxed);
            }
        };
        let context = context.clone();

        tokio::select! {
            res = cam_fut => res,
            res = peer_fut => res,
            res = spawn_blocking(|| self.network_node.run(context)) => res.unwrap(),
        }
    }
}
