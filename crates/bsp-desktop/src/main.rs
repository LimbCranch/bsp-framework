//! BSP Desktop Application - Simple EMG Visualization

mod app;
mod ui;

use app::BSPApp;
use tracing_subscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Starting BSP-Framework Desktop Application...");

    // Configure egui
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    // Create and run the application
    eframe::run_native(
        "BSP-Framework - EMG Simulator",
        options,
        Box::new(|_cc| {
            // Create the app inside the closure where we have a tokio runtime available
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            let app = rt.block_on(async {
                BSPApp::new().expect("Failed to create BSP app")
            });
            Ok(Box::new(app))
        }),
    ).map_err(|e| format!("Failed to run native app: {}", e))?;

    Ok(())
}