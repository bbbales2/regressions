
use pyo3::prelude::*;
use pyo3::prelude::{PyResult, Python, PyObject};
use pyo3::types::IntoPyDict;
use pyo3::conversion::ToPyObject;
use ndarray::*;
use numpy::{IntoPyArray, PyReadonlyArray1};
use std::iter::FromIterator;
use numpy::ToPyArray;
use pyo3::{pymodule, types::PyModule};

fn uturn(q_plus: &ArrayView1<f64>, q_minus: &ArrayView1<f64>, p_plus: &ArrayView1<f64>, p_minus: &ArrayView1<f64>) -> bool {
    let uturn_forward = (q_plus - q_minus).dot(p_plus) <= 0.0;
    let uturn_backward = -(q_minus - q_plus).dot(p_minus) <= 0.0;

    uturn_forward && uturn_backward
}

fn kinetic_energy(p : &Array1<f64>, diag_m_inv : &Array1<f64>) -> f64 {
    0.5 * (p * diag_m_inv).dot(p)
}

fn value_and_grad(py: Python, py_value_and_grad: &PyObject, q : &Array1<f64>) -> (f64, Array1<f64>) {
    let res = py_value_and_grad.call1(py, (q.to_pyarray(py),)).unwrap().extract(py);
    let (u, grad) : (f64, Vec<f64>) = match res {
        Ok(v) => v,
        Err(_) => panic!("Error evaluating gradient (calling value_and_grad from Rust)")
    };
    (u, Array1::from_vec(grad))
}

fn log_sum_exp(x : &Vec<f64>) -> f64 {
    let y = x.iter().fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));
    y + x.iter().map(|xv| (xv - y).exp()).sum::<f64>().ln()
}
  
fn softmax(x : &Vec<f64>) -> Vec<f64>{
    Vec::from_iter(x.iter().map(|xv| (xv - log_sum_exp(x)).exp()))
}

#[pyfunction]
fn do_it<'py>(
    py: Python<'py>,
    py_value_and_grad : PyObject,
    py_rng : PyObject,
    py_q0 : PyReadonlyArray1<f64>,
    py_diagonal_inverse_metric : PyReadonlyArray1<f64>,
    stepsize : f64,
    max_treedepth : usize
) -> PyResult<Vec<PyObject>> {
    let q0 = py_q0.as_array().to_owned();
    let diag_m_inv = py_diagonal_inverse_metric.as_array().to_owned();

    let size = q0.len();
    let kwargs = [("size", size)].into_py_dict(py);
    let z : Array1<f64> = Array1::from_vec(py_rng.call_method(py, "normal", (), Some(kwargs)).unwrap().extract(py)?);
    let kwargs = [("low", 0), ("high", 2), ("size", max_treedepth + 1)].into_py_dict(py);
    let directions : Vec<i32> = py_rng.call_method(py, "integers", (), Some(kwargs)).unwrap().extract(py)?;
    let kwargs = [("size", max_treedepth)].into_py_dict(py);
    let choice_draws : Vec<f64> = py_rng.call_method(py, "random", (), Some(kwargs)).unwrap().extract(py)?;
    let pick_draws : Vec<f64> = py_rng.call_method(py, "random", (), Some(kwargs)).unwrap().extract(py)?;

    // Allocate return values
    let mut q_ret = Array1::zeros(size);
    let mut accept_stat : Option<f64> = None;
    let mut total_leapfrogs_taken = 0;
    let mut divergence = false;

    // Apparently using as for the conversion is kinda bad cause it can mess up
    let max_leapfrogs : usize = 2_usize.pow(max_treedepth as u32);

    // To compute the U-turn conditions, we need to save our positions/momentums
    // Technically we only need to save the new subtree we computed, but to avoid
    // a bunch of mallocs we just save the whole trajectory

    // qs stores our positions
    let mut qs = Array2::zeros((max_leapfrogs, size));
    // ps stores our momentums
    let mut ps = Array2::zeros((max_leapfrogs, size));

    py.allow_threads(||{
        let p0 = z / diag_m_inv.mapv(|x : f64| x.sqrt());//Vec::from_iter(z.iter().zip(diag_m_inv.iter()).map(|(z, m_inv)| z / m_inv.sqrt()));

        let (u0, _) = Python::with_gil(|py| {
            value_and_grad(py, &py_value_and_grad, &q0)
        });
        let h0 = kinetic_energy(&p0, &diag_m_inv) + u0;

        // directions is a length max_treedepth vector that maps each treedepth
        //  to an integration direction (left or right)
        let directions : Vec<i32> = Vec::from_iter(directions.iter().map(|d| if *d == 0 {-1} else {1}));
            
        // Compute the position of the initial condition in the trajectory
        let mut i_first = 0;
        for (depth, &direction) in directions[1..].iter().enumerate() {
            if direction < 0 {
                i_first = i_first + (1 << depth);
            }
        }

        // log_pi defined in section A.2.3 of https://arxiv.org/abs/1701.02434
        let mut log_pi = vec![0_f64; max_leapfrogs];

        qs.row_mut(i_first).assign(&q0);
        ps.row_mut(i_first).assign(&p0);
        log_pi[i_first] = -h0;
        
        // i_left and i_right are indices that track the leftmost and rightmost
        //  states of the integrated trajectory
        let mut i_left = i_first;
        let mut i_right = i_first;
        // log_sum_pi_old is the log of the sum of the pis (of log_pi) for the
        //  tree processed so far
        let mut log_sum_pi_old = log_pi[i_first];
        // i_old will be the sample chosen (the sample from T(z' | told) in section
        //  A.3.1 of https://arxiv.org/abs/1701.02434)
        let mut i_old = i_first;
        // For trees of increasing treedepth
        
        let mut checks = Vec::new();
        checks.reserve(2 << max_treedepth);
        for depth in 1..(max_treedepth + 1) {
            // Figure out what leapfrog steps we need to compute. If integrating in the
            //  positive direction update the index that points at the right side of the
            //  trajectory. If integrating in the negative direction update the index pointing
            //  to the left side of the trajectory.
            let max_leapfrog_steps = 1 << (depth - 1);
            let depth_steps = if directions[depth] < 0 {
                Vec::from_iter(((i_left - max_leapfrog_steps)..i_left).rev())
            } else {
                Vec::from_iter((i_right + 1)..(i_right + 1 + max_leapfrog_steps))
            };
            
            let i_prev;
            if directions[depth] < 0 {
                i_prev = i_left;
                i_left = *depth_steps.last().unwrap();
            } else {
                i_prev = i_right;
                i_right = *depth_steps.last().unwrap();
            }

            // What we're doing here is generating a trajectory composed of a number of leapfrog states.
            // We apply a set of comparisons on this trajectory that can be organized to look like a binary tree.
            // Sometimes I say trajectory instead of tree. Each piece of the tree in the comparison corresponds
            //  to a subset of the trajectory. When I say trajectory I'm referring to a piece of the trajectory that
            //  also corresponds to some sorta subtree in the comparisons.
            //
            // This is probably confusing but what I want to communicate is trajectory and tree are very related
            //  but maybe technically not the same thing.
            //
            // Detect U-turns in newly integrated subtree
            let mut uturn_detected_new_tree = false;

            // Starts and ends are relative because they point to the ith leapfrog step in a sub-trajectory
            //  of size 2^(depth - 1) (this is max_leapfrog_steps) which needs to be mapped to the global
            // index of qs (which is defined in the range 1:2^max_treedepth)
            //
            // We're assuming at this point that depth_steps is in sorted order
            //
            // The sort is necessary because depth_steps is sorted in order of leapfrog steps taken
            //  which might be backwards in time (so decreasing instead of increasing)
            //
            // This sort is important because we need to keep track of what is left and what is right
            //  in the trajectory so that we do the right comparisons
            if depth > 1 {
                // Each new subtree is twice as wide as the previous. For each half of the new
                // subtree, we need to do all the checks as the previous subtree, plus one
                // additional end-to-end check
                let div_length = max_leapfrog_steps >> 1;
                // Add the start of left to start of right check
                checks.push((0, div_length));
                // Copy and shift the previous checks
                for i in 0..checks.len() - 1 {
                    let (x, y) = checks[i];
                    checks.push((x + div_length, y + div_length));
                }
                // Add the end of left to end of right check
                checks.push((div_length - 1, 2 * div_length - 1));
                // And the additional start of left to end of right check
                checks.push((0, 2 * div_length - 1));
            }

            // Sort the checks so that we only need to check them in order
            checks.sort_by_key(|&(_, end)| end);

            let dt = stepsize * directions[depth] as f64;

            let mut q = qs.row(i_prev).to_owned();
            let mut p = ps.row(i_prev).to_owned();

            let (_, qgrad) = Python::with_gil(|py| {
                value_and_grad(py, &py_value_and_grad, &q)
            });
            let mut grad = qgrad;

            // Initialize pointer into checks list
            let mut check_i = 0;

            let mut leapfrogs_taken = 0;
            for i in depth_steps.iter() {
                // leapfrog step
                leapfrogs_taken += 1;

                let p_half = p - (dt / 2.0) * grad;
                q = q + dt * &diag_m_inv * &p_half;
                let (u, qgrad) = Python::with_gil(|py| {
                    value_and_grad(py, &py_value_and_grad, &q)
                });
                grad = qgrad;
                p = p_half - (dt / 2.0) * &grad;

                let k = kinetic_energy(&p, &diag_m_inv);
                let h = k + u;

                qs.row_mut(*i).assign(&q);
                ps.row_mut(*i).assign(&p);

                log_pi[*i] = -h;

                while check_i < checks.len() {
                    let (left, right) = checks[check_i];
                    if right < leapfrogs_taken {
                        let (start, end) = if directions[depth] > 0 { (left, right) } else { (depth_steps.len() - left - 1, depth_steps.len() - right - 1) };
                        let start_i = depth_steps[start];
                        let end_i = depth_steps[end];
    
                        let is_uturn = uturn(
                            &qs.row(end_i),
                            &qs.row(start_i),
                            &ps.row(end_i),
                            &ps.row(start_i),
                        );
    
                        if is_uturn {
                            uturn_detected_new_tree = true;
                            break;
                        }

                        check_i += 1;
                    } else {
                        break;
                    }
                }

                if uturn_detected_new_tree {
                    break;
                }
            }

            total_leapfrogs_taken += leapfrogs_taken;

            // We need to know whether we terminated the trajectory cause of a uturn or we
            //  hit the max trajectory length
            // Merging the two trees requires one more uturn check from the overall left to right states
            let uturn_detected = uturn_detected_new_tree || uturn(&qs.row(i_right), &qs.row(i_left), &ps.row(i_right), &ps.row(i_left));

            // Accept statistic from ordinary HMC
            // Only compute the accept probability for the steps done
            let mut p_tree_accept = 0.0;
            let log_pi_steps = Vec::from_iter((0..leapfrogs_taken).map(|i| log_pi[depth_steps[i]]));
            for i in 0..leapfrogs_taken {
                let energy_loss = h0 + log_pi_steps[i];
                p_tree_accept += energy_loss.exp().min(1.0) / leapfrogs_taken as f64;

                if energy_loss.is_nan() || energy_loss.abs() > 1000.0 {
                    divergence = true;
                }
            }

            // Divergence
            if divergence {
                if accept_stat.is_none() {
                    accept_stat = Some(0.0);
                }

                break;
            }

            if uturn_detected {
                // If we u-turn on the first step, grab something for the accept_stat
                if accept_stat.is_none() {
                    accept_stat = Some(p_tree_accept);
                }

                break
            }

            accept_stat = match accept_stat {
                None => Some(p_tree_accept),
                Some(accept_stat) => {
                    let weight = leapfrogs_taken as f64 / (leapfrogs_taken + depth_steps.len() - 1) as f64;
                    Some(accept_stat * (1.0 - weight) + p_tree_accept * weight)
                }
            };

            // log of the sum of pi (A.3.1 of https://arxiv.org/abs/1701.02434) of the new subtree
            let log_sum_pi_new = log_sum_exp(&log_pi_steps);

            // sample from the new subtree according to the equation in A.2.1 in https://arxiv.org/abs/1701.02434
            //  (near the end of that section)
            let i_new = if depth > 1 {
                let probs = softmax(&log_pi_steps);

                let choice_draw = choice_draws[depth - 1];

                let mut total_prob = 0.0_f64;
                let mut depth_step : usize = 0;
                for i in 0..probs.len() {
                    let next_prob = total_prob + probs[i];
                    depth_step = depth_steps[i];
                    if choice_draw >= total_prob && choice_draw < next_prob {
                        break;
                    } else {
                        total_prob = next_prob;
                    }
                }
                depth_step
            } else {
                depth_steps[0]
            };

            // Pick between the samples generated from the new and old subtrees using the biased progressive sampling in
            //  A.3.2 of https://arxiv.org/abs/1701.02434
            let p_new = (log_sum_pi_new - log_sum_pi_old).exp().min(1.0);
            if pick_draws[depth - 1] < p_new {
                i_old = i_new
            }

            // Update log of sum of pi of overall tree
            log_sum_pi_old = log_sum_exp(&vec![log_sum_pi_old, log_sum_pi_new]);

        }

        // Get the final sample
        q_ret.assign(&qs.row(i_old));
    });

    Ok(vec![
        q_ret.into_pyarray(py).to_object(py),
        accept_stat.unwrap().to_object(py),
        total_leapfrogs_taken.to_object(py),
        divergence.to_object(py)
    ])
}

#[pymodule]
fn one_draw(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(do_it, m)?)?;
    Ok(())
}