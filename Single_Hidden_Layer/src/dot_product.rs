pub fn apply_dot_product(a : &mut Vec<Vec<f64>>, b : &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_cols = b[0].len();

    let mut z: Vec<Vec<f64>> = vec![vec![0.0; b_cols]; a_rows];

    for i in 0..a_rows {

        for j in 0..b_cols {

            let mut sum: f64 = 0.0;
            for k in 0..a_cols {

                sum += a[i][k] * b[k][j];
            }

            z[i][j] = sum;
        }
    }

    return z;
}

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}