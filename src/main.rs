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
    \"sigma_tr\"   : [<macroscopic xsec in cm^-1 for group 1>, <same for group 2>],
    \"sigma_a\"    : [<group 1>, <group 2>],
    \"nu_sigma_f\" : [<group 1>, <group 2>],
    \"chi\"        : [<probability to be created in group 1>, <group 2>],
    \"sigma_s\"    : [[<macroscopic xsec in cm^-1 to scatter from group 1 to group 1>, <group 1 to group 2>],
                  [<group 2 to group 1>, <group 2 to group 2>]]
}
";

const GROUPS: usize = 2;

#[derive(Serialize, Deserialize)]
struct Material {
    sigma_tr: [f64; GROUPS], // TODO: make this n-group
    sigma_a: [f64; GROUPS],
    nu_sigma_f: [f64; GROUPS],
    chi: [f64; GROUPS],
    sigma_s: [[f64; GROUPS]; GROUPS]
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
    let core_segments = (parameters.core_thickness / delta).round() as u64;
    let reflector_segments = (parameters.reflector_thickness / delta).round() as u64;

    assert!(core_segments + 2 * reflector_segments == parameters.num_segments, "num_segments doesn't evenly divide the core and reflectors.");

    // Generate material data vectors
    // let sigma_a_core = Array::from_elem(
}
