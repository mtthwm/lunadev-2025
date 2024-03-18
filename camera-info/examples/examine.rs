use std::io::stdin;

use camera::{discover_all_cameras, Camera};
use camera_info::interactive_examine;
use unros::{
    anyhow::{self, Context}, tokio::{self, task::JoinHandle}, Application
};



#[unros::main]
async fn main(mut app: Application) -> anyhow::Result<Application> {
    discover_all_cameras()
        .context("Failed to discover cameras")?
        .for_each(|_| {});

    let join: JoinHandle<Result<_, anyhow::Error>> = tokio::task::spawn_blocking(|| {
        let stdin = stdin();
        println!("Please provide a camera index");
        let mut input = String::new();
        let index = loop {
            stdin.read_line(&mut input)?;
            let Ok(index) = input.trim().parse::<u32>() else {
                println!("Invalid integer!");
                input.clear();
                continue;
            };
        
            break index;
        };

        Ok(index)
    });
    
    let index = join.await.unwrap()?;

    let mut camera = Camera::new(index)?;
    let camera_name = camera.get_camera_name().to_string();
    interactive_examine(&mut app, |x| camera.accept_image_received_sub(x), camera_name).await;
    app.add_node(camera);

    Ok(app)
}
