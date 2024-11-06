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
        help = "Optional, should the detection results be displayed in the gui window, default is false"
    )]
    show: bool,
    #[arg(
        short,
        long,
        help = "Optional, multiple category names, each category separated directly by commas"
    )]
    names: Option<String>,
    #[arg(
        short,
        long,
        default_value = "0.5",
        help = "Optional, confidence threshold for detection results"
    )]
    threshold: f32,
}

//Default object detect names with COCO
const DEFAULT_NAMES: &str = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    let mut config = ModelRunConfig::default();
    config.yolo_version = args.yolo_version;
    config.threshold = args.threshold;
    if let Some(names) = args.names {
        config.names = names.split(",").map(|v| v.trim().to_string()).collect();
    } else {
        config.names = DEFAULT_NAMES
            .split(",")
            .map(|v| v.trim().to_string())
            .collect();
    }

    let runner = ModelRunner::new(args.model.as_str(), config).unwrap();

    if args.show {
        highgui::named_window("window", highgui::WINDOW_KEEPRATIO)?;
        highgui::resize_window("window", 720, 480)?;
    }

    runner.run(args.input.as_str(), ModelRunner::no_pre, |res, mut mat| {
        if args.show {
            show_result_image(&res, &mut mat);
        }
        println!("Result: {:?}", &res);
    })?;
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
