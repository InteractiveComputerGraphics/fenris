use fenris::quadrature::univariate::gauss;
use fenris::quadrature::{OwnedQuadratureParts, Quadrature};
use itertools::izip;
use nalgebra::Point1;

mod canonical;
mod subdivide;

#[test]
fn quadrature_iter() {
    let quadrature = {
        let base_quadrature = gauss::<f64>(4);
        let size = base_quadrature.0.len();
        let data = vec![2.0; size];
        OwnedQuadratureParts::from(base_quadrature).with_data(data)
    };
    let q = &quadrature;

    let quadrature_izip_collected: Vec<(&f64, &Point1<f64>, &f64)> = izip!(q.weights(), q.points(), q.data()).collect();
    let quadrature_iter_collected: Vec<(&f64, &Point1<f64>, &f64)> = q.iter().collect();

    assert_eq!(quadrature_iter_collected, quadrature_izip_collected);
}
