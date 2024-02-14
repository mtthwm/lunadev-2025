use std::{
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use crossbeam::queue::SegQueue;
use enet::{
    Address, BandwidthLimit, ChannelLimit, Enet, Event, Host, Packet, PacketMode, PeerState,
};
use global_msgs::Steering;
use image::DynamicImage;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use ordered_float::NotNan;
use rand::seq::SliceRandom;
use unros_core::{
    anyhow, async_trait, log,
    pubsub::{Publisher, Subscriber, Subscription},
    rayon::{
        self,
        iter::{IntoParallelIterator, ParallelIterator},
    },
    rng::QuickRng,
    setup_logging, tokio_rayon, Node, RuntimeContext,
};

#[derive(Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
enum Channels {
    Important,
    Camera,
    Odometry,
    Controls,
    Max,
}

#[derive(Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
enum ImportantMessage {
    EnableCamera,
    DisableCamera,
    Ping,
}

/// A remote connection to `Lunabase`
pub struct Telemetry {
    pub bandwidth_limit: u32,
    pub server_addr: Address,
    pub max_image_chunk_width: u32,
    steering_signal: Publisher<Steering>,
    image_subscriptions: Subscriber<Arc<DynamicImage>>,
    packet_queue: SegQueue<(Box<[u8]>, PacketMode, Channels)>,
}

impl Telemetry {
    pub fn new(server_addr: impl Into<Address>) -> Self {
        Self {
            bandwidth_limit: 0,
            server_addr: server_addr.into(),
            steering_signal: Default::default(),
            image_subscriptions: Subscriber::new(1),
            packet_queue: SegQueue::new(),
            max_image_chunk_width: 32,
        }
    }

    pub fn accept_steering_sub(&mut self, sub: Subscription<Steering>) {
        self.steering_signal.accept_subscription(sub);
    }

    pub fn create_image_subscription(&self) -> Subscription<Arc<DynamicImage>> {
        self.image_subscriptions.create_subscription()
    }

    fn receive_packet(&mut self, channel: u8, packet: Box<[u8]>, context: &RuntimeContext) {
        setup_logging!(context);
        let Ok(channel) = Channels::try_from(channel) else {
            error!("Received invalid channel: {channel}");
            return;
        };
        match channel {
            Channels::Important => {
                let Ok(msg) = ImportantMessage::try_from(packet[0]) else {
                    error!("Received invalid ImportantMessage: {}", packet[0]);
                    return;
                };
                match msg {
                    ImportantMessage::EnableCamera => todo!(),
                    ImportantMessage::DisableCamera => todo!(),
                    ImportantMessage::Ping => self.packet_queue.push((
                        packet,
                        PacketMode::ReliableSequenced,
                        Channels::Important,
                    )),
                }
            }
            Channels::Camera => todo!(),
            Channels::Odometry => todo!(),
            Channels::Controls => {
                let drive = i8::from_le_bytes([packet[0]]) as f32;
                let steering = i8::from_le_bytes([packet[1]]) as f32;

                self.steering_signal.set(Steering::from_drive_and_steering(
                    NotNan::new(drive / 127.0).unwrap(),
                    NotNan::new(steering / 127.0).unwrap(),
                ));

                self.packet_queue.push((
                    packet,
                    PacketMode::UnreliableUnsequenced,
                    Channels::Controls,
                ));
            }
            Channels::Max => error!("Received invalid channel: {}", channel as u8),
        }
    }
}

struct HostWrapper(Host<()>);

impl Deref for HostWrapper {
    type Target = Host<()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HostWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for HostWrapper {
    fn drop(&mut self) {
        let mut peer_count = 0;
        for mut peer in self.0.peers() {
            if peer.state() == PeerState::Connected {
                peer.disconnect(0);
                peer_count += 1;
            }
        }
        while peer_count > 0 {
            let mut event = match self.0.service(1000) {
                Ok(Some(event)) => event,
                Ok(None) => continue,
                Err(e) => {
                    log::error!("Faced the following error while safely dropping ENet host: {e}");
                    break;
                }
            };
            match &mut event {
                Event::Connect(peer) => {
                    peer_count += 1;
                    peer.disconnect(0);
                }
                Event::Disconnect(_, _) => peer_count -= 1,
                _ => {}
            }
        }
    }
}

struct DropCheck(Arc<AtomicBool>, std::sync::mpsc::Receiver<()>);

impl Drop for DropCheck {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
        let _ = self.1.recv();
    }
}

#[async_trait]
impl Node for Telemetry {
    const DEFAULT_NAME: &'static str = "telemetry";

    async fn run(mut self, context: RuntimeContext) -> anyhow::Result<()> {
        setup_logging!(context);

        let enet = Enet::new()?;
        let outgoing_limit = if self.bandwidth_limit == 0 {
            BandwidthLimit::Unlimited
        } else {
            BandwidthLimit::Limited(self.bandwidth_limit)
        };

        let drop_check_bool = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = std::sync::mpsc::sync_channel(0);
        let _drop_check = DropCheck(drop_check_bool.clone(), receiver);

        tokio_rayon::spawn(move || {
            let _sender = sender;

            loop {
                let host = enet.create_host::<()>(
                    None,
                    1,
                    ChannelLimit::Maximum,
                    BandwidthLimit::Unlimited,
                    outgoing_limit,
                )?;
                let mut host = HostWrapper(host);
                host.connect(&self.server_addr, Channels::Max as usize, 0)?;
                info!("Connecting to lunabase...");
                loop {
                    if drop_check_bool.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let Some(event) = host.service(50)? else {
                        continue;
                    };
                    match event {
                        Event::Connect(_) => break,
                        Event::Disconnect(_, _) => {
                            warn!("Somehow disconnected from a peer!")
                        }
                        Event::Receive { .. } => todo!(),
                    }
                }

                info!("Connected to lunabase!");
                let mut start_service = Instant::now();
                loop {
                    {
                        let option = host.service(50)?;
                        if drop_check_bool.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        match option {
                            Some(Event::Connect(_)) => {
                                error!("Somehow connected to another peer, ignoring...");
                            }
                            Some(Event::Disconnect(_, _)) => {
                                error!("Disconnected from lunabase!");
                                break;
                            }
                            Some(Event::Receive {
                                channel_id,
                                ref packet,
                                ..
                            }) => {
                                let packet = packet.data().to_vec().into_boxed_slice();
                                self.receive_packet(channel_id, packet, &context);
                            }
                            None => {}
                        }
                    }
                    let mut peer = host.peers().next().unwrap();
                    while let Some((body, mode, channel)) = self.packet_queue.pop() {
                        peer.send_packet(Packet::new(&body, mode)?, channel as u8)?;
                    }
                    let elapsed = start_service.elapsed();
                    if elapsed.as_millis() < 50 {
                        continue;
                    }
                    start_service += elapsed;
                    if let Some(img) = self.image_subscriptions.try_recv() {
                        let w_chunks = img.width().div_ceil(self.max_image_chunk_width) as u16;
                        let h_chunks = img.height().div_ceil(self.max_image_chunk_width) as u16;
                        let mut rng = QuickRng::default();

                        let mut xy_vec: Vec<_> = (0..w_chunks)
                            .flat_map(|x| (0..h_chunks).map(move |y| (x, y)))
                            .collect();
                        xy_vec.shuffle(&mut rng);

                        let (sender, recv) = std::sync::mpsc::sync_channel(xy_vec.len());

                        rayon::spawn(move || {
                            xy_vec.into_par_iter().for_each(move |(x, y)| {
                                let mut chunk_width;
                                if x == w_chunks - 1 {
                                    chunk_width = img.width() % self.max_image_chunk_width;
                                    if chunk_width == 0 {
                                        chunk_width = self.max_image_chunk_width;
                                    }
                                } else {
                                    chunk_width = self.max_image_chunk_width;
                                }

                                let mut chunk_height;

                                if y == h_chunks - 1 {
                                    chunk_height = img.height() % self.max_image_chunk_width;
                                    if chunk_height == 0 {
                                        chunk_height = img.height() % self.max_image_chunk_width;
                                    }
                                } else {
                                    chunk_height = self.max_image_chunk_width;
                                }

                                let img = webp::Encoder::from_image(&img.crop_imm(
                                    x as u32 * self.max_image_chunk_width,
                                    y as u32 * self.max_image_chunk_width,
                                    chunk_width,
                                    chunk_height,
                                ))
                                .unwrap()
                                .encode(10.0);
                                let mut bytes = Vec::with_capacity(4 + img.len());

                                bytes.extend_from_slice(&x.to_le_bytes());
                                bytes.extend_from_slice(&y.to_le_bytes());
                                bytes.extend_from_slice(&img);

                                sender.send(bytes).unwrap();
                            });
                        });
                        for bytes in recv {
                            peer.send_packet(
                                Packet::new(&bytes, enet::PacketMode::UnreliableUnsequenced)?,
                                Channels::Camera as u8,
                            )?;
                        }
                    }
                }
            }
        })
        .await
    }
}
