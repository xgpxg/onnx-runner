use std::fs::OpenOptions;
use std::io::Write;
use std::process::exit;
use std::str::FromStr;
use std::time::Duration;

use clap::Parser;
use eyre::ErrReport;
use opencv::core::{Mat, Point, Rect, Scalar};
use opencv::highgui;
use opencv::imgproc::HersheyFonts;
use reqwest::blocking::Client;
use serde_json::json;
use thiserror::Error;

use crate::runner::runner::{ModelRunConfig, ModelRunResult, ModelRunner};

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
    #[arg(
        short,
        long,
        help = "Optional, send results to the specified location. Send to file: file://your_path/your_file, send yo http(s) api: http://host/path"
    )]
    output: Option<String>,
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

    let runner = ModelRunner::new(args.model.as_str(), config)?;

    if args.show {
        highgui::named_window("window", highgui::WINDOW_KEEPRATIO)?;
        highgui::resize_window("window", 720, 480)?;
    }

    let mut output = Output::from_str(args.output.unwrap_or_default().as_str())?;

    runner.run(args.input.as_str(), ModelRunner::no_pre, |res, mut mat| {
        println!("Result: {:?}", &res);

        if args.show {
            show_result_image(&res, &mut mat);
        }

        let _ = send_result(&mut output, &res).map_err(|why| {
            eprintln!("Send result fail: {}", why.to_string());
        });
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

#[derive(Debug)]
enum Output {
    None,
    FILE(String),
    HTTP(String, Client),
    HTTPS(String, Client),
}

#[derive(Error, Debug)]
pub enum OutputError {
    #[error(
        "Not supported output: {0}. The output must be start with file:// or http:// or https://"
    )]
    NotSupportedError(String),
}

impl FromStr for Output {
    type Err = ErrReport;

    fn from_str(output: &str) -> eyre::Result<Self, ErrReport> {
        if output == "" {
            return Ok(Output::None);
        }
        if output.starts_with("file://") {
            let path = output.replace("file://", "");
            Ok(Output::FILE(path))
        } else if output.starts_with("http://") {
            let client = get_client()?;
            Ok(Output::HTTP(output.to_string(), client))
        } else if output.starts_with("https://") {
            let client = get_client()?;
            Ok(Output::HTTPS(output.to_string(), client))
        } else {
            eprintln!(
                "Error: {}",
                OutputError::NotSupportedError(output.to_string())
            );
            exit(1);
        }
    }
}

fn get_client() -> eyre::Result<Client> {
    let client = Client::builder()
        .pool_max_idle_per_host(10)
        .connect_timeout(Duration::from_secs(30))
        .build()?;
    Ok(client)
}

fn send_result(output: &mut Output, res: &Vec<ModelRunResult>) -> eyre::Result<()> {
    match output {
        Output::None => {
            //Nothing to do
        }
        Output::FILE(file) => {
            let mut file = OpenOptions::new().append(true).open(file)?;
            writeln!(file, "{}", json!(res))?;
        }
        Output::HTTP(url, client) => {
            println!("Sending result to: {}", url);
            client.post(url.as_str()).body("").send()?;
            let response = client
                .post(url.as_str())
                .body(json!(res).to_string())
                .send()?;
            println!(
                "Server response: <{}>\n{}",
                response.status(),
                response.text()?
            );
        }
        Output::HTTPS(url, client) => {
            println!("Sending result to: {}", url);
            let response = client
                .post(url.as_str())
                .body(json!(res).to_string())
                .send()?;
            println!(
                "Server response: <{}>\n{}",
                response.status(),
                response.text()?
            );
        }
    }
    Ok(())
}
