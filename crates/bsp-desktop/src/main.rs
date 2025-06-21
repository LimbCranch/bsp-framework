//! BSP Desktop Application - Complete EMG Processing Pipeline

mod app;
mod ui;
mod processing_service;

use app::BSPApp;
use tracing_subscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Starting BSP-Framework Desktop Application...");
    println!("Signal Flow: EMG Simulator → Processing Pipeline → Real-time Visualization");

    // Configure egui
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([1000.0, 700.0]),
        ..Default::default()
    };

    // Create and run the application
    eframe::run_native(
        "BSP-Framework - Real-time EMG Processing",
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