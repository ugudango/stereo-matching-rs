use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI32, Ordering};
use opencv::core::{CV_8U, Scalar, Size, Vector};
use opencv::imgcodecs::{imread, IMREAD_GRAYSCALE, imwrite};
use opencv::prelude::*;
use opencv::highgui::imshow;
use opencv::highgui::wait_key;
use opencv::imgcodecs::ImwriteFlags::IMWRITE_PNG_BILEVEL;
use rayon::prelude::*;

struct Stereo {
    kernel_size: i32,
    max_disparity: i32,
    tran_size: i32,
}

fn hamming_distance(a: i32, b: i32) -> i32 {
    let (mut d, mut res) = (a ^ b, 0);
    while d > 0 {
        res += d & 1;
        d >>= 1;
    }
    res
}

struct MatWrapper(Mat);

unsafe impl Sync for MatWrapper {}

impl Stereo {
    pub fn census_transform(image: Mat, kernel_size: i32) -> Mat {
        let (h, w) = (image.rows(), image.cols());

        let mut img_disparity: Mat = Mat::new_size_with_default(
            Size::new(w, h),
            CV_8U,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        ).unwrap();

        let half_kernel_size = kernel_size / 2;

        for x in half_kernel_size..(h - half_kernel_size) {
            for y in half_kernel_size..(w - half_kernel_size) {
                let mut ssd: u8 = 0;

                for u in -half_kernel_size..(half_kernel_size + 1) {
                    for v in -half_kernel_size..(half_kernel_size + 1) {
                        if (u, v) != (0, 0) {
                            if image.at_2d::<u8>(x + u, y + v).unwrap() >= image.at_2d::<u8>(x, y).unwrap() {
                                ssd |= 1;
                            }
                            ssd <<= 1;
                        }
                    }
                }

                *img_disparity.at_2d_mut::<u8>(x, y).unwrap() = ssd;
            }
        }

        println!("Census transform done!");
        img_disparity
    }

    pub fn stereo_match(&self, mut left: Mat, mut right: Mat) -> Mat {
        let (h, w) = (left.rows(), left.cols());
        let mut img_disparity = Mat::new_size_with_default(
            Size { width: w, height: h },
            CV_8U,
            Scalar::from(0),
        ).unwrap();

        let k_half = self.kernel_size;
        let adjust = 255 / self.max_disparity;

        let left_imm = Arc::new(Mutex::new(MatWrapper(left.clone())));
        let right_imm = Arc::new(Mutex::new(MatWrapper(right.clone())));

        let images_in = vec!(left_imm, right_imm);
        let mut images_out = vec!();

        images_in.into_par_iter().map(|mat| {
            let locked_mat = mat.lock().unwrap();
            MatWrapper(Self::census_transform(locked_mat.0.clone(), self.tran_size))
        }).collect_into_vec(&mut images_out);

        let mut img_disparity_arc = Arc::new(Mutex::new(img_disparity));

        (k_half..h - k_half).into_par_iter().for_each(|i| {
            for j in k_half..w - k_half {
                let mut prev_ssd = i32::MAX;
                let mut best_disparity = 0;

                for offset in 0..self.max_disparity {
                    let mut ssd = 0;
                    for u in -k_half..k_half {
                        for v in -k_half..k_half {
                            let ssd_tmp = hamming_distance(
                                *images_out[0].0.at_2d::<u8>(i + u, j + v).unwrap() as i32,
                                *images_out[1].0.clone().at_2d::<u8>(i + u, (j + v - offset).clamp(0, i32::MAX)).unwrap() as i32,
                            );

                            ssd += ssd_tmp * ssd_tmp
                        };
                    }

                    if ssd < prev_ssd {
                        prev_ssd = ssd;
                        best_disparity = offset;
                    }
                }
                let mut img_disp = img_disparity_arc.lock().unwrap();
                *img_disp.at_2d_mut::<u8>(i, j).unwrap() = (best_disparity * adjust) as u8;
            }
        });

        let mut img_disp = img_disparity_arc.lock().unwrap();
        img_disp.clone()
    }
}

fn main() {
    let left = imread("./im2.png", IMREAD_GRAYSCALE).unwrap();
    let right = imread("./im6.png", IMREAD_GRAYSCALE).unwrap();

    let s = Stereo {
        kernel_size: 3,
        max_disparity: 50,
        tran_size: 7,
    };

    imwrite("stereo.png", &s.stereo_match(left, right), &Vector::<i32>::new());

    wait_key(0);
}
