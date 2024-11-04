use crate::runner::runner::{ModelRunConfig, ModelRunResult, ModelRunner};
use clap::Parser;
use eyre::OptionExt;
use opencv::core::{Mat, Point, Rect, Scalar};
use opencv::highgui;
use opencv::imgproc::HersheyFonts;

mod runner;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        help = "YOLO onnx model file path, support version: v5, v7, v8, v10, and v11"
    )]
    model: String,
    #[arg(
        short,
        long,
        help = "Input source, like image file, http image, camera, or rtsp"
    )]
    input: String,
    #[arg(
        long,
        default_value = "8",
        help = "The number of YOLO version, like 5, 7 ,8 ,10, or 11. Specifically, for YOLO 10, it needs to be set up"
    )]
    yolo_version: usize,
    #[arg(
        long,
        default_value = "false",
        help = "Should the detection results be displayed in the gui window, default is false"
    )]
    show: bool,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    let mut config = ModelRunConfig::default();
    config.yolo_version = args.yolo_version;
    let runner = ModelRunner::new(args.model.as_str(), config).unwrap();

    if args.show {
        highgui::named_window("window", highgui::WINDOW_KEEPRATIO)?;
        highgui::resize_window("window", 720, 480)?;
    }

    runner.run(
        args.input.as_str(),
        ModelRunner::no_pre,
        |res, mut mat| {
            if args.show {
                show_result_image(&res, &mut mat);
            }
            println!("Result: {:?}", &res);
        },
    )?;
    if args.show {
        highgui::wait_key(-1)?;
    }
    Ok(())
}

fn show_result_image(res: &Vec<ModelRunResult>, mat: &mut Mat) {
    for item in res {
        let _ = opencv::imgproc::rectangle(
            mat,
            Rect::from_points(
                Point::new(item.bounding_box.x1 as i32, item.bounding_box.y1 as i32),
                Point::new(item.bounding_box.x2 as i32, item.bounding_box.y2 as i32),
            ),
            Scalar::new(0., 0., 255., 1.),
            2,
            8,
            0,
        );
        let _ = opencv::imgproc::put_text(
            mat,
            item.label,
            Point::new(item.bounding_box.x1 as i32, item.bounding_box.y1 as i32),
            HersheyFonts::FONT_HERSHEY_PLAIN.into(),
            1.,
            Scalar::new(0., 0., 255., 1.),
            1,
            8,
            false,
        );
    }
    highgui::imshow("window", mat).unwrap();
    highgui::wait_key(1).unwrap();
}