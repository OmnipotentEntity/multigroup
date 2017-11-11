extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate serde_derive;

extern crate ndarray;

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::env;
use std::iter::Iterator;
use ndarray::{Array1, Array2, Array3, Axis, stack};

// This program is intended to solve a 1D slab reactor with a reflector.
// Using 2 group techniques

const USAGE: &str = "
Usage: multigroup <parameters.json> <core.json> <reflector.json>

<parameters.json> is expect to be a path to JSON data of the following format:
{
    \"core_thickness\" : core_thickness in cm,
    \"reflector_thickness\" : reflector_thickness in cm on either side of the core,
    \"num_segments\" : region thickness in cm
}

###==========###
Ref   Core   Ref

<core.json> and <reflector.json> are expected to be paths to JSON data of the following format:

{
    \"sigma_tr\"   : {
      \"v\" : 1,
      \"dim\" : [<number of groups>],
      \"data\" : [<comma separated list of macro xsec for each group, starting with group 1>]
    },
    \"sigma_r\"    : {
      \"v\" : 1,
      \"dim\" : [<number of groups>],
      \"data\" : [<comma separated list of macro xsec for each group, starting with group 1>]
    },
    \"nu_sigma_f\" : {
      \"v\" : 1,
      \"dim\" : [<number of groups>],
      \"data\" : [<comma separated list of macro xsec for each group, starting with group 1>]
    },
    \"chi\"        : {
      \"v\" : 1,
      \"dim\" : [<number of groups>],
      \"data\" : [<comma separated list of probabilities for each group, starting with group 1>]
    },
    \"sigma_s\"    : {
      \"v\" : 1,
      \"dim\" : [<number of groups>, <number of groups>],
      \"data\" : [<flattened matrix of macro xsec starting with group 1 to 1, then group 1 to 2 and so on>]
    }
}
";

#[derive(Serialize, Deserialize)]
struct Material {
    sigma_tr: Array1<f64>,
    sigma_r: Array1<f64>,
    nu_sigma_f: Array1<f64>,
    chi: Array1<f64>,
    sigma_s: Array2<f64>
}

#[derive(Serialize, Deserialize)]
struct Parameters {
    core_thickness: f64,
    reflector_thickness: f64,
    num_segments: usize
}

// Reads json data from a file and parses it into a Material struct
fn read_mat_from_file<P: AsRef<Path>>(path: P) -> Result <Material, Box<Error>> {
    // Open the file in read-only mode.
    let file = File::open(path)?;

    // Read the JSON contents of the file as a instance of Material.
    let mat = serde_json::from_reader(file)?;

    // Return the Material
    Ok(mat)
}

// Reads json data from a file and parses it into a Parameters struct
fn read_par_from_file<P: AsRef<Path>>(path: P) -> Result <Parameters, Box<Error>> {
    // Open the file in read-only mode.
    let file = File::open(path)?;

    // Read the JSON contents of the file as a instance of Parameters.
    let par = serde_json::from_reader(file)?;

    // Return the Parameters
    Ok(par)
}

// This function exists to not repeat functional code
fn make_array_from_mix_impl(core: &f64, refl: &f64, gr: &mut f64, refl_seg: f64, core_seg: f64, i: usize) {
    // NOTE: There is a limitation of the below.  It doesn't correctly calculate percentages on
    // the very first or very last section.  I could put it in, but then the code would be even
    // messier and it's an edge case.

    // If we're completely in the domain of the reflector
    if i < (refl_seg.round() as usize) {
        *gr = *refl;
        // If we're in an overlap region between the reflector and the core
    } else if i == (refl_seg.round() as usize) {
        let reflp = refl_seg - (i as f64) + 0.5;
        *gr = refl * reflp + core * (1.0 - reflp);
        // If we're in the core
    } else if i < ((refl_seg + core_seg).round() as usize) {
        *gr = *core;
        // If we're in the second overlap region
    } else if i == ((refl_seg + core_seg).round() as usize) {
        let corep = refl_seg + core_seg - (i as f64) + 0.5;
        *gr = core * corep + refl * (1.0 - corep);
        // Else, we're in the second reflector
    } else {
        *gr = *refl;
    }
}

// This function uses the parameters and the material to create a spacial distribution of a
// specific physical property, each row is a single group.
fn make_array2_from_mix(core: &Array1<f64>, refl: &Array1<f64>, parameters: &Parameters) -> Array2<f64> {
    let total_thickness = parameters.core_thickness + 2.0 * parameters.reflector_thickness;
    let delta = total_thickness / (parameters.num_segments as f64);
    let core_segments = parameters.core_thickness / delta;
    let reflector_segments = parameters.reflector_thickness / delta;

    // TODO: This is messy as hell, there has to be a better way to construct this array. :/
    let mut res = Array2::<f64>::zeros((0, parameters.num_segments));
    for (cgroup, rgroup) in core.iter().zip(refl) {
        let mut group_res = Array2::<f64>::zeros((1, parameters.num_segments));
        for (i, gr) in group_res.iter_mut().enumerate() {
            make_array_from_mix_impl(cgroup, rgroup, gr, reflector_segments, core_segments, i);
        }
        res = stack(Axis(0), &[res.view(), group_res.view()]).unwrap();
    }

    res
}

fn make_array3_from_mix(core: &Array2<f64>, refl: &Array2<f64>, parameters: &Parameters) -> Array3<f64> {
    let total_thickness = parameters.core_thickness + 2.0 * parameters.reflector_thickness;
    let delta = total_thickness / (parameters.num_segments as f64);
    let core_segments = parameters.core_thickness / delta;
    let reflector_segments = parameters.reflector_thickness / delta;

    // TODO: This is messy as hell, there has to be a better way to construct this array. :/
    let mut res = Array3::<f64>::zeros((0, core.len_of(Axis(1)), parameters.num_segments));
    for (core_vec, refl_vec) in core.axis_iter(Axis(0)).zip(refl.axis_iter(Axis(0))) {
        let mut row_res = Array3::<f64>::zeros((1, 0, parameters.num_segments));
        for (cgroup, rgroup) in core_vec.iter().zip(refl_vec)  {
            let mut col_res = Array3::<f64>::zeros((1, 1, parameters.num_segments));
            for (i, gr) in col_res.iter_mut().enumerate() {
                make_array_from_mix_impl(cgroup, rgroup, gr, reflector_segments, core_segments, i);
            }
            row_res = stack(Axis(1), &[row_res.view(), col_res.view()]).unwrap();
        }
        res = stack(Axis(0), &[res.view(), row_res.view()]).unwrap();
    }

    res
}

fn main() {
    // Read data from files
    let parameters_filename = env::args().nth(1).expect(USAGE);
    let core_filename = env::args().nth(2).expect(USAGE);
    let reflector_filename = env::args().nth(3).expect(USAGE);

    let parameters = read_par_from_file(&parameters_filename).unwrap();
    let core = read_mat_from_file(&core_filename).unwrap();
    let reflector = read_mat_from_file(&reflector_filename).unwrap();

    // Double check data for sanity
    assert!(parameters.core_thickness > 0.0, "Core thickness must be positive.");
    assert!(parameters.reflector_thickness > 0.0, "Reflector thickness must be positive.");
    assert!(core.sigma_tr.len() == core.sigma_r.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == core.nu_sigma_f.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == core.chi.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == core.sigma_s.len_of(Axis(0)), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == core.sigma_s.len_of(Axis(1)), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == reflector.sigma_tr.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == reflector.sigma_r.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == reflector.nu_sigma_f.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == reflector.chi.len(), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == reflector.sigma_s.len_of(Axis(0)), "Number of groups must be the same for all xsecs");
    assert!(core.sigma_tr.len() == reflector.sigma_s.len_of(Axis(1)), "Number of groups must be the same for all xsecs");

    // Simple derived quantities
    let total_thickness = parameters.core_thickness + 2.0 * parameters.reflector_thickness;
    let delta = total_thickness / (parameters.num_segments as f64);

    // Generate material data vectors
    let sigma_tr   = make_array2_from_mix(&core.sigma_tr,   &reflector.sigma_tr,   &parameters);
    let sigma_r    = make_array2_from_mix(&core.sigma_r,    &reflector.sigma_r,    &parameters);
    let nu_sigma_f = make_array2_from_mix(&core.nu_sigma_f, &reflector.nu_sigma_f, &parameters);
    let chi        = make_array2_from_mix(&core.chi,        &reflector.chi,        &parameters);
    let sigma_s    = make_array3_from_mix(&core.sigma_s,    &reflector.sigma_s,    &parameters);

    println!("sigma_tr: {}", sigma_tr);
}
