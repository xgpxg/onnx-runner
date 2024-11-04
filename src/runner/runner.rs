use std::str::FromStr;

use ndarray::{s, Axis};
use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, Size, CV_32F};
use opencv::prelude::VideoCaptureTrait;
use opencv::videoio;
use opencv::videoio::VideoCapture;
use ort::{
    CUDAExecutionProvider, GraphOptimizationLevel, Session, SessionOutputs,
    Tensor, TensorRTExecutionProvider,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub struct ModelRunner {
    pub id: String,
    pub session: Session,
    pub config: ModelRunConfig,
    pub close_flag: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

pub struct ModelRunConfig {
    pub resize: (i32, i32),
    pub names: Vec<String>,
    pub threshold: f32,
    pub yolo_version: usize,
    pub input_param_name: String,
    pub output_param_name: String,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelRunResult<'a> {
    pub bounding_box: BoundingBox,
    pub label: &'a str,
    pub prob: f32,
}

impl Default for ModelRunConfig {
    fn default() -> Self {
        let default_names = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(",").map(|v| v.trim().to_string()).collect();
        ModelRunConfig {
            resize: (640, 640),
            names: default_names,
            threshold: 0.5,
            yolo_version: 8,
            input_param_name: "images".to_string(),
            output_param_name: "output0".to_string(),
        }
    }
}

impl ModelRunner {
    ///Create a new model runner with model file and model config
    ///
    /// `model_path` onnx model file path
    ///
    /// `config` model config
    pub fn new(model_path: &str, config: ModelRunConfig) -> eyre::Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                TensorRTExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        Ok(ModelRunner {
            id: Uuid::new_v4().to_string(),
            config,
            session,
            close_flag: false,
        })
    }

    fn run_with_mat(&self, image: &Mat) -> eyre::Result<Vec<ModelRunResult>> {
        let config = &self.config;
        let session = &self.session;
        let input = image_to_onnx_input(image, config.resize)?;
        let outputs = session.run(ort::inputs![&config.input_param_name => input]?)?;
        let result = convert_result(&outputs, image, config)?;
        Ok(result)
    }

    ///Run model with input
    ///
    /// `input` Support image file, http/https, camera, rtsp
    ///
    /// `p` Preprocessing input image
    ///
    /// `f` Model run result callback function, contains detect result and after preprocessing image(the return value of function `p`)
    pub fn run<P, F>(&self, input: &str, p: P, f: F) -> eyre::Result<()>
    where
        P: Fn(Mat) -> Mat,
        F: Fn(Vec<ModelRunResult>, Mat),
    {
        let mut capture = get_video_capture(input)?;
        let mut frame = Mat::default();
        while !self.close_flag {
            let success = capture.read(&mut frame)?;
            if !success {
                break;
            }
            frame = p(frame);
            let result = self.run_with_mat(&frame)?;
            f(result, frame.clone());
        }
        capture.release()?;
        Ok(())
    }

    pub fn no_pre(mat: Mat) -> Mat {
        mat
    }

    ///Stop runner
    pub fn stop(&mut self) -> eyre::Result<()> {
        self.close_flag = true;
        self.session.end_profiling()?;
        Ok(())
    }
}

const SCHEMA_CAMERA: &str = "camera://";

fn get_video_capture(input: &str) -> eyre::Result<VideoCapture> {
    if input.starts_with(SCHEMA_CAMERA) {
        Ok(VideoCapture::new(
            input.replace(SCHEMA_CAMERA, "").parse::<i32>()?,
            videoio::CAP_ANY,
        )?)
    } else {
        Ok(VideoCapture::from_file(input, videoio::CAP_ANY)?)
    }
}

pub fn image_to_onnx_input(image: &Mat, resize: (i32, i32)) -> eyre::Result<Tensor<f32>> {
    let mat = opencv::dnn::blob_from_image(
        image,
        1. / 255.,
        Size::new(resize.0, resize.1),
        Default::default(),
        true,
        false,
        CV_32F,
    )?;

    let array = Tensor::from_array(([1, 3, resize.0, resize.1], mat.data_typed::<f32>().unwrap()))?;
    Ok(array)
}

pub fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

pub fn convert_result<'a>(
    outputs: &SessionOutputs,
    original_image: &Mat,
    config: &'a ModelRunConfig,
) -> eyre::Result<Vec<ModelRunResult<'a>>> {
    let (img_width, img_height) = (original_image.cols(), original_image.rows());
    let output_name = config.output_param_name.as_str();
    let output = outputs[output_name]
        .try_extract_tensor::<f32>()?
        .t()
        .into_owned();

    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    match config.yolo_version {
        10 => {
            for row in output.axis_iter(Axis(1)) {
                let row: Vec<_> = row.iter().copied().collect();
                let prob = row[4];
                if prob < config.threshold {
                    continue;
                }
                let class_id = row[5] as usize;
                let label = config.names[class_id].as_str();

                let bounding_box = BoundingBox {
                    x1: row[0] / config.resize.0 as f32 * (img_width as f32),
                    y1: row[1] / config.resize.1 as f32 * (img_height as f32),
                    x2: row[2] / config.resize.0 as f32 * (img_width as f32),
                    y2: row[3] / config.resize.1 as f32 * (img_height as f32),
                };

                boxes.push(ModelRunResult {
                    bounding_box,
                    label,
                    prob,
                });
            }
        }
        _ => {
            for row in output.axis_iter(Axis(0)) {
                let row: Vec<_> = row.iter().copied().collect();
                let (class_id, prob) = row
                    .iter()
                    // skip bounding box coordinates
                    .skip(4)
                    .enumerate()
                    .map(|(index, value)| (index, *value))
                    .reduce(|accum, e| if e.1 > accum.1 { e } else { accum })
                    .unwrap();
                if prob < config.threshold {
                    continue;
                }
                let label = config.names[class_id].as_str();

                let xc = row[0] / config.resize.0 as f32 * (img_width as f32);
                let yc = row[1] / config.resize.1 as f32 * (img_height as f32);
                let w = row[2] / config.resize.0 as f32 * (img_width as f32);
                let h = row[3] / config.resize.1 as f32 * (img_height as f32);

                let bounding_box = BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.,
                };
                boxes.push(ModelRunResult {
                    bounding_box,
                    label,
                    prob,
                });
            }
        }
    }

    boxes.sort_by(|box1, box2| box2.prob.total_cmp(&box1.prob));
    let mut result = Vec::new();

    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| {
                intersection(&boxes[0].bounding_box, &box1.bounding_box)
                    / union(&boxes[0].bounding_box, &box1.bounding_box)
                    < 0.7
            })
            .copied()
            .collect();
    }

    Ok(result)
}
