use eframe::{egui, NativeOptions};
use morphnet_gtl::prelude::*;
use morphnet_gtl::VERSION;

struct MorphNetApp {
    net: MorphNet,
}

impl eframe::App for MorphNetApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("MorphNet GUI");
            ui.label(format!("MorphNet version: {}", VERSION));
            ui.label(format!("Templates loaded: {}", self.net.body_plan.templates.len()));
        });
    }
}

fn main() -> eframe::Result<()> {
    let app = MorphNetApp { net: MorphNet::new() };
    eframe::run_native(
        "MorphNet GUI",
        NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
}
