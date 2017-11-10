extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate serde_derive;

extern crate ndarray;

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::env;

// This program is intended to solve a 1D slab reactor with a reflector.
// Using 2 group techniques

const USAGE: &'static str = "
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
    \"sigma_a\"    : {
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
    sigma_tr: ndarray::Array1<f64>,
    sigma_a: ndarray::Array1<f64>,
    nu_sigma_f: ndarray::Array1<f64>,
    chi: ndarray::Array1<f64>,
    sigma_s: ndarray::Array2<f64>
}

#[derive(Serialize, Deserialize)]
struct Parameters {
    core_thickness: f64,
    reflector_thickness: f64,
    num_segments: u64
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

// This function uses the parameters and the material 
// fn make_vector_from_mix<P: AsRef<Parameters>>(mat_core: &[f64; 2], mat_refl: &[f64; 2], parameters: P) -> ndarray::Array2<f64> {
//     let group1 = Array1::
// }

fn main() {
    // Read data from files
    let parameters_filename = env::args().nth(1).expect(USAGE);
    let core_filename = env::args().nth(2).expect(USAGE);
    let reflector_filename = env::args().nth(3).expect(USAGE);

    let parameters = read_par_from_file(&parameters_filename).unwrap();
    let core = read_mat_from_file(&core_filename).unwrap();
    let reflector = read_mat_from_file(&reflector_filename).unwrap();

    // Simple derived quantities
    let total_thickness = parameters.core_thickness + 2.0 * parameters.reflector_thickness;
    let delta = total_thickness / (parameters.num_segments as f64);

    // Generate material data vectors
    // let sigma_a_core = Array::from_elem(
    

}
