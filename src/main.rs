extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::env;
use std::iter::Iterator;
use ndarray::{Array, Array1, Array2, Array3, ArrayView, Axis, Shape, stack};

// This program is intended to solve a 1D slab reactor with a reflector.
// Using 2 group techniques

const GROUPERR: &str = "Number of groups must be the same for all xsecs";
const COREERR: &str = "Core thickness must be positive.";
const REFLERR: &str = "Reflector thickness must be positive.";

const USAGE: &str = "
Usage: multigroup <parameters.json> <core.json> <reflector.json>

<parameters.json> is expect to be a path to JSON data of the following format:
{
    \"source_convergence\" : ratio for source convergence criteria,
    \"criticality_convergence\" : ratio for criticality convergence criteria,
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
    sigma_tr: Array1<f64>,
    sigma_a: Array1<f64>,
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

struct ReactorParameters {
    sigma_tr: Array2<f64>,
    sigma_a: Array2<f64>,
    sigma_r: Array2<f64>,
    nu_sigma_f: Array2<f64>,
    chi: Array2<f64>,
    sigma_s: Array3<f64>
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
    if (i as f64) + 0.5 < refl_seg.round() + 0.5 {
        *gr = *refl;
        // If we're in an overlap region between the reflector and the core
    } else if (i as f64) + 0.5 == refl_seg.round() + 0.5 {
        let reflp = refl_seg - (i as f64) + 0.5;
        *gr = refl * reflp + core * (1.0 - reflp);
        // If we're in the core
    } else if (i as f64) + 0.5 < (refl_seg + core_seg).round() + 0.5 {
        *gr = *core;
        // If we're in the second overlap region
    } else if (i as f64) + 0.5 == (refl_seg + core_seg).round() + 0.5 {
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
    let delta = total_thickness / ((parameters.num_segments - 1) as f64);
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
    let delta = total_thickness / ((parameters.num_segments - 1) as f64);
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

fn gen_sigma_r(sigma_a: &Array2<f64>, sigma_s: &Array3<f64>) -> Array2<f64> {
    // \Sigma_r = \Sigma_a + \sum\limits_{g' \ne g} \Sigma_s

    Array::from_shape_fn(sigma_a.dim(),
        |(i, j)| sigma_a[(i, j)] +
                 sigma_s.slice(s![i as isize..(i+1) as isize, .., j as isize..(j+1) as isize])
                    .fold(0.0, |sum, curr| sum + curr) - sigma_s[(i, i, j)])
}

fn update_source_from_flux(reactor: &ReactorParameters, flux: &Array2<f64>, source: &mut Array2<f64>, delta: f64) {
    // The formula for this is in LaTeX format is:
    // S_i = \frac1{k} \chi_g \sum\limits_{g'} \frac1{8} \Delta \left( 3 \phi_i \left( \nu_{g'}
    // \Sigma_{fg',i-1} + \nu_{g'} \Sigma_{fg',i} \right) + \nu_{g'} \Sigma_{fg',i-1}
    // \phi_{i-1} + \nu_{g'} \Sigma_{fg',i} \phi_{i+1} \right)

    // S_0 = \frac1{k} \chi_g \sum\limits_{g'} \frac1{8} \Delta \left( 3 \phi_i \nu_{g'}
    // \Sigma_{fg',i} + \nu_{g'} \Sigma_{fg',i} \phi_{i+1} \right)

    // S_N = \frac1{k} \chi_g \sum\limits_{g'} \frac1{8} \Delta \left( 3 \phi_i \nu_{g'}
    // \Sigma_{fg',i-1} + \nu_{g'} \Sigma_{fg',i-1} \phi_{i-1} \right)

    let groups = source.len_of(Axis(0));
    let segments: usize = source.strides()[0] as usize;

    for (i, loc_source) in source.iter_mut().enumerate() {
        let left_contrib: f64;
        let segment_id = i % segments;
        let group_id = i / segments;
        if (segment_id != 0) {
            // S_N only has left contrib, so this is essentially the S_N formula
            let mut sum = 0.0;
            for group in (0..groups) {
                sum += 3.0 * flux[[group, segment_id]]     * reactor.nu_sigma_f[[group, segment_id - 1]]
                           + flux[[group, segment_id - 1]] * reactor.nu_sigma_f[[group, segment_id - 1]];
            }
            left_contrib = sum * delta / 8.0;

        } else {
            left_contrib = 0.0;
        }

        let right_contrib: f64;
        if (segment_id != segments - 1) {
            // S_0 only has left contrib, so this is essentially the S_N formula
            let mut sum = 0.0;
            for group in (0..groups) {
                sum += 3.0 * flux[[group, segment_id]] * reactor.nu_sigma_f[[group, segment_id]]
                           + flux[[group, segment_id]] * reactor.nu_sigma_f[[group, segment_id + 1]];
            }
            right_contrib = sum * delta / 8.0;

        } else {
            right_contrib = 0.0;
        }

        // And then we just add up the contributions, and multiply by the group constant. The 1/k
        // constant will need to be multiplied later, as it must be calculated using this source.
        *loc_source = reactor.chi[[group_id, segment_id]] * (left_contrib + right_contrib);
    }
}

fn make_flux_from_source(reactor: &ReactorParameters, flux: &Array2<f64>, source: &Array2<f64>, delta: f64) -> (f64, Array2<f64>) {
    // The formula for this in LaTeX format is:
    // \left(\frac{D_i + D_{i-1}}{\Delta}\right)\phi_i - \frac{D_{i-1}}{\Delta}\phi_{i-1} -
    // \frac{D_i}{\Delta}\phi_{i+1} + \frac1{8}\Delta\left(3\phi_i \left(\Sigma_{r,i-1} +
    // \Sigma_{r,i}\right)\right) + \Sigma_{r,i-1}\phi_{i-1} + \Sigma_{r,i}\phi_{i+1} = \left(
    // \frac{S_{i-1} + S_i}{2} \right) \Delta + \sum\limits_{g' \ne g} \left(\frac1{8} \Delta
    // \left(3 \phi_{g',i} \left(\Sigma_{sg'g,i-1} + \Sigma_{sg'g,i}\right)\right) +
    // \Sigma_{sg'g,i-1}\phi_{g',i-1} + \Sigma_{sg'g,i}\phi_{g',i+1}\right)

    // Where S_i is the source given by the previous formula.

    // This formula is already split from left to right with thing that depend on the current value
    // of \phi in this group.  Meaning it is possible to solve this system of equations for the
    // current value of \phi.

    // However, if the reactor is super critical or sub critical we get an exponential flux
    // increase or decrease.  So at the end we total up the old flux and new flux and renormalize.
    // This value is the inverse of the criticality, and this is of interest so we also return it.

    let mut res = flux.clone();
    let crit = 1.0;

    let groups = source.len_of(Axis(0));
    let segments: usize = source.strides()[0] as usize;

    (crit, res)
}

fn make_group_matrices(reactor: &ReactorParameters) -> Array3<f64> {
    // In order to solve the system of equations we'll be solving it using a matrix inverse.  So
    // first we need to construct the matrix.
    // [[ b_0 c_0   0 ...   0]
    //  [ a_1 b_1 c_1 ...   0]
    //  [   0 a_2 b_2 ...   0]
    //  ...
    //  [   0   0   0 ... b_N]]

    // Where:
    // a_i = \frac1{8}\Delta\Sigma_{r,i-1} - \frac{D_{i-1}}{\Delta}
    // b_i = \frac{3}{8}\Delta(\Sigma{r,i-1} + \Sigma{r,i}) + \frac{D_{i-1} + D_i}{\Delta}
    // c_i = \frac1{8}\Delta\Sigma_{r,i} - \frac{D_{i}}{\Delta}
    // Each being the components multiplied by \phi_{i-1}, \phi{i} and \phi{i+1} respectively.

    let mut res: Array3<f64> = Array::zeros(
        (reactor.sigma_tr.len_of(Axis(0)), 
         reactor.sigma_tr.len_of(Axis(0)), 
         reactor.sigma_tr.len_of(Axis(1))));

    for group in res.axis_iter_mut(Axis(2)) {
        // group[[0,0]] = reactor.
        // for i in (0..group.len_of(Axis(0))) {

        // }
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
    assert!(parameters.core_thickness > 0.0, COREERR);
    assert!(parameters.reflector_thickness > 0.0, REFLERR);
    assert!(core.sigma_tr.len() == core.sigma_a.len(), GROUPERR);
    assert!(core.sigma_tr.len() == core.nu_sigma_f.len(), GROUPERR);
    assert!(core.sigma_tr.len() == core.chi.len(), GROUPERR);
    assert!(core.sigma_tr.len() == core.sigma_s.len_of(Axis(0)), GROUPERR);
    assert!(core.sigma_tr.len() == core.sigma_s.len_of(Axis(1)), GROUPERR);
    assert!(core.sigma_tr.len() == reflector.sigma_tr.len(), GROUPERR);
    assert!(core.sigma_tr.len() == reflector.sigma_a.len(), GROUPERR);
    assert!(core.sigma_tr.len() == reflector.nu_sigma_f.len(), GROUPERR);
    assert!(core.sigma_tr.len() == reflector.chi.len(), GROUPERR);
    assert!(core.sigma_tr.len() == reflector.sigma_s.len_of(Axis(0)), GROUPERR);
    assert!(core.sigma_tr.len() == reflector.sigma_s.len_of(Axis(1)), GROUPERR);

    // Simple derived quantities
    let total_thickness = parameters.core_thickness + 2.0 * parameters.reflector_thickness;
    let delta = total_thickness / ((parameters.num_segments - 1) as f64);

    // Generate material data vectors
    let sigma_tr   = make_array2_from_mix(&core.sigma_tr,   &reflector.sigma_tr,   &parameters);
    let sigma_a    = make_array2_from_mix(&core.sigma_a,    &reflector.sigma_a,    &parameters);
    let nu_sigma_f = make_array2_from_mix(&core.nu_sigma_f, &reflector.nu_sigma_f, &parameters);
    let chi        = make_array2_from_mix(&core.chi,        &reflector.chi,        &parameters);
    let sigma_s    = make_array3_from_mix(&core.sigma_s,    &reflector.sigma_s,    &parameters);
    let sigma_r    = gen_sigma_r(&sigma_a, &sigma_s);
    let arr3shape = (sigma_tr.len_of(Axis(0)), sigma_tr.len_of(Axis(1)), 1);

    // Generate flat source and flux data

    let mut criticality_history = Array::from_elem((1), 1.0);
    let mut source_history = Array::from_elem(arr3shape, 1.0);
    let mut flux_history   = Array::from_elem(arr3shape, 1.0);

    let mut source: Array2<f64> = Array::from_elem(sigma_tr.dim(), 1.0);
    let mut flux:   Array2<f64> = Array::from_elem(sigma_tr.dim(), 1.0);

    let mut iterations = 0;

    let reactor = ReactorParameters{sigma_tr, sigma_a, sigma_r, nu_sigma_f, chi, sigma_s};

    // Finally a "do-while" loop in order to update source and flux until they converge
    // loop {
    //     // First, update the source from the flux
    //     update_source_from_flux(&reactor, &flux, &mut source, delta);
    //     let source_to_save = source.clone().into_shape(arr3shape).unwrap();
    //     source_history = stack(Axis(2), &[source_history.view(), source_to_save.view()]).unwrap();

    //     // Next, update the flux from the source.
    //     // This one does not modify in place because we have to reference flux from other groups
    //     // in order to calculate the flux of a single group.  So we have to keep all data around
    //     // until we're completely done with it.
    //     let (criticality, new_flux) = make_flux_from_source(&reactor, &flux, &source, delta);
    //     let new_flux_to_save = flux.clone().into_shape(arr3shape).unwrap();
    //     flux_history = stack(Axis(2), &[flux_history.view(),new_flux_to_save.view()]).unwrap();
    //     flux = new_flux;
    // }
}
