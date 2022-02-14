use crate::{ConvexPolygon, Disk, HalfPlane, Plane3d};
use nalgebra::{clamp, Matrix2, RealField, Vector2};
use nalgebra::{Point2, Point3, Scalar};
use numeric_literals::replace_float_literals;
use std::fmt::Debug;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineSegment3d<T: Scalar> {
    end_points: [Point3<T>; 2],
}

impl<T: Scalar> LineSegment3d<T> {
    pub fn from_end_points(end_points: [Point3<T>; 2]) -> Self {
        Self { end_points }
    }
}

impl<T: RealField> LineSegment3d<T> {
    pub fn project_point_parametric(&self, point: &Point3<T>) -> T {
        let a = self.end_points[0].coords;
        let b = self.end_points[1].coords;
        let d = &b - &a;
        let d2 = d.magnitude_squared();
        if d2 == T::zero() {
            // If the endpoints are the same, the segment collapses to a single point,
            // in which case e.g. t == 0 gives the correct solution.
            T::zero()
        } else {
            let x = point.coords;
            let t = (x - &a).dot(&d) / d2;
            t
        }
    }

    pub fn project_point(&self, point: &Point3<T>) -> Point3<T> {
        let t = self.project_point_parametric(point);
        if t <= T::zero() {
            self.end_points[0]
        } else if t >= T::one() {
            self.end_points[1]
        } else {
            self.point_from_parameter(t)
        }
    }

    pub fn point_from_parameter(&self, t: T) -> Point3<T> {
        let a = self.end_points[0];
        let b = self.end_points[1];
        Point3::from(a.coords * (T::one() - t) + &b.coords * t)
    }

    #[allow(non_snake_case)]
    pub fn closest_point_to_plane_parametric(&self, plane: &Plane3d<T>) -> T {
        let n = plane.normal();
        let x0 = plane.point();
        let [a, b] = &self.end_points;
        let d = &b.coords - &a.coords;
        let y = &x0.coords - &a.coords;

        let nTd = n.dot(&d);
        let nTy = n.dot(&y);

        // The parameter t is generally given by the equation
        //  dot(n, d) * t = dot(n, y)
        // but we must be careful, since dot(n, d) can get arbitrarily close to 0,
        // which causes some challenges.
        let t = if nTd.signum() == nTy.signum() {
            // Sign is the same, thus t >= 0
            if nTy.abs() >= nTd.abs() {
                T::one()
            } else {
                nTy / nTd
            }
        } else {
            // t must be negative, directly clamp to zero
            T::zero()
        };

        t
    }

    pub fn closest_point_to_plane(&self, plane: &Plane3d<T>) -> Point3<T> {
        let t = self.closest_point_to_plane_parametric(plane);
        self.point_from_parameter(t)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LineSegment2d<T>
where
    T: Scalar,
{
    start: Point2<T>,
    end: Point2<T>,
}

impl<T> LineSegment2d<T>
where
    T: Scalar,
{
    pub fn new(from: Point2<T>, to: Point2<T>) -> Self {
        Self { start: from, end: to }
    }

    pub fn start(&self) -> &Point2<T> {
        &self.start
    }

    pub fn end(&self) -> &Point2<T> {
        &self.end
    }

    pub fn reverse(&self) -> Self {
        LineSegment2d {
            start: self.end.clone(),
            end: self.start.clone(),
        }
    }
}

impl<T> LineSegment2d<T>
where
    T: RealField,
{
    pub fn to_line(&self) -> Line2d<T> {
        let dir = &self.end - &self.start;
        Line2d::from_point_and_dir(self.start.clone(), dir)
    }

    /// Returns a vector tangent to the line segment.
    ///
    /// Note that the vector is **not** normalized.
    pub fn tangent_dir(&self) -> Vector2<T> {
        self.end().coords - self.start().coords
    }

    /// Returns a vector normal to the line segment, in the direction consistent with a
    /// counter-clockwise winding order when the edge is part of a polygon.
    ///
    /// Note that the vector is **not** normalized.
    pub fn normal_dir(&self) -> Vector2<T> {
        let tangent = self.tangent_dir();
        Vector2::new(tangent.y, -tangent.x)
    }

    pub fn length(&self) -> T {
        self.tangent_dir().norm()
    }

    pub fn midpoint(&self) -> Point2<T> {
        Point2::from((self.start.coords + self.end.coords) / (T::one() + T::one()))
    }

    pub fn intersect_line_parametric(&self, line: &Line2d<T>) -> Option<T> {
        self.to_line()
            .intersect_line_parametric(line)
            .map(|(t1, _)| t1)
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_disk(&self, disk: &Disk<T>) -> Option<Self> {
        let [t1, t2] = self.to_line().intersect_disk_parametric(disk)?;
        let t1 = clamp(t1, 0.0, 1.0);
        let t2 = clamp(t2, 0.0, 1.0);
        Some(self.segment_from_parameters(&t1, &t2))
    }

    /// Compute the closest point on the segment to the given point, represented in
    /// the parametric form x = a + t * (b - a).
    pub fn closest_point_parametric(&self, point: &Point2<T>) -> T {
        let t = self.to_line().project_point_parametric(point);
        clamp(t, T::zero(), T::one())
    }

    /// Computes the closest point on the line to the given point.
    pub fn closest_point(&self, point: &Point2<T>) -> Point2<T> {
        let t = self.closest_point_parametric(point);
        self.point_from_parameter(t)
    }

    pub fn point_from_parameter(&self, t: T) -> Point2<T> {
        Point2::from(self.start().coords + (self.end() - self.start()) * t)
    }

    pub fn segment_from_parameters(&self, t_begin: &T, t_end: &T) -> Self {
        let begin = self.point_from_parameter(t_begin.clone());
        let end = self.point_from_parameter(t_end.clone());
        Self::new(begin, end)
    }

    /// Computes the intersection of two line segments (if any), but returns the result as a parameter.
    ///
    /// Let all points on this line segment be defined by the relation x = a + t * (b - a)
    /// for 0 <= t <= 1. Then, if the two line segments intersect, t is returned. Otherwise,
    /// `None` is returned.
    pub fn intersect_segment_parametric(&self, other: &LineSegment2d<T>) -> Option<T> {
        // Represent the two lines as:
        //  x1 = a1 + t1 * d1
        //  x2 = a2 + t2 * d2
        // where di = bi - ai. This gives the linear system
        //  [ d1  -d2 ] t = a2 - a1,
        // where t = [t1, t2].

        let d1 = &self.end - &self.start;
        let d2 = &other.end - &other.start;

        let line1 = Line2d::from_point_and_dir(self.start.clone(), d1);
        let line2 = Line2d::from_point_and_dir(other.start.clone(), d2);

        line1
            .intersect_line_parametric(&line2)
            .and_then(|(t1, t2)| {
                // TODO: This may go very wrong if we're talking "exact" intersection
                // e.g. when a line segment intersects another segment only at a point,
                // in which case we might discard the intersection entirely.
                // I suppose the only way to deal with this is either arbitrary precision
                // or using epsilons? Also, keep in mind that the `from` and `to`
                // points may already be suffering from imprecision!
                if t2 < T::zero() || t2 > T::one() {
                    None
                } else if t1 < T::zero() || t1 > T::one() {
                    None
                } else {
                    Some(t1)
                }
            })
    }

    /// Compute the intersection between the line segment and a half-plane.
    ///
    /// Returns `None` if the segment and the half-plane do not intersect, otherwise
    /// returns `Some([t1, t2])` with `t1 <= t2`, and `t1` and `t2` correspond to the start and end parameters
    /// relative to the current line segment.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_half_plane_parametric(&self, half_plane: &HalfPlane<T>) -> Option<[T; 2]> {
        let contains_start = half_plane.contains_point(self.start());
        let contains_end = half_plane.contains_point(self.end());

        match (contains_start, contains_end) {
            (true, true) => Some([0.0, 1.0]),
            (false, false) => None,
            (true, false) | (false, true) => {
                let t_intersect = self
                    .intersect_line_parametric(&half_plane.surface())
                    // Technically the intersection should be in the interval [0, 1] already,
                    // but numerical errors may lead to values that are slightly outside, or, in the case of
                    // very nearly parallel lines, far outside.
                    .map(|t| clamp(t, 0.0, 1.0));

                let (t_start, t_end);
                if contains_start {
                    // The only case when the intersection returns None is when the half-plane line and the
                    // line segment are parallel, which we *technically* have excluded already.
                    // But due to floating-point imprecision we might still find ourselves in this situation.
                    // In this case the result may be more or less arbitrary, so we pick a reasonable default
                    // to fall back on
                    t_start = 0.0;
                    t_end = t_intersect.unwrap_or(1.0);
                } else {
                    t_start = t_intersect.unwrap_or(0.0);
                    t_end = 1.0;
                }

                debug_assert!(t_start <= t_end);
                Some([t_start, t_end])
            }
        }
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_half_plane(&self, half_plane: &HalfPlane<T>) -> Option<Self> {
        self.intersect_half_plane_parametric(half_plane)
            .map(|[t1, t2]| self.segment_from_parameters(&t1, &t2))
    }

    pub fn intersect_polygon(&self, other: &ConvexPolygon<T>) -> Option<LineSegment2d<T>> {
        let mut result = self.clone();
        for half_plane in other.half_planes() {
            result = result.intersect_half_plane(&half_plane)?;
        }
        Some(result)
    }
}

#[derive(Debug, Clone)]
pub struct Line2d<T>
where
    T: Scalar,
{
    point: Point2<T>,
    dir: Vector2<T>,
}

impl<T> Line2d<T>
where
    T: Scalar,
{
    pub fn from_point_and_dir(point: Point2<T>, dir: Vector2<T>) -> Self {
        // TODO: Make dir Unit?
        Self { point, dir }
    }

    pub fn point(&self) -> &Point2<T> {
        &self.point
    }

    pub fn dir(&self) -> &Vector2<T> {
        &self.dir
    }
}

impl<T> Line2d<T>
where
    T: RealField,
{
    /// A normalized vector tangent to the line.
    pub fn tangent(&self) -> Vector2<T> {
        self.dir.normalize()
    }

    pub fn from_point_through_point(point: Point2<T>, through: &Point2<T>) -> Self {
        let dir = through - &point;
        Self::from_point_and_dir(point, dir)
    }

    /// Computes the projection of the given point onto the line, representing the point
    /// in parametric form.
    pub fn project_point_parametric(&self, point: &Point2<T>) -> T {
        let d2 = self.dir.magnitude_squared();
        (point - &self.point).dot(&self.dir) / d2
    }

    /// Computes the projection of the given point onto the line.
    pub fn project_point(&self, point: &Point2<T>) -> Point2<T> {
        let t = self.project_point_parametric(point);
        self.point_from_parameter(t)
    }

    pub fn point_from_parameter(&self, t: T) -> Point2<T> {
        &self.point + &self.dir * t
    }

    pub fn intersect(&self, other: &Line2d<T>) -> Option<Point2<T>> {
        self.intersect_line_parametric(other)
            .map(|(t1, _)| self.point_from_parameter(t1))
    }

    /// Computes the intersection of two lines, if any.
    ///
    /// Let all points on each line segment be defined by the relation `x1 = a1 + t1 * d1`
    /// for `0 <= t1 <= 1`, and similarly for `t2`. Here, `t1` is the parameter associated with
    /// `self`, and `t2` is the parameter associated with `other`.
    pub fn intersect_line_parametric(&self, other: &Line2d<T>) -> Option<(T, T)> {
        // Represent the two lines as:
        //  x1 = a1 + t1 * d1
        //  x2 = a2 + t2 * d2
        // where di = bi - ai. This gives the linear system
        //  [ d1  -d2 ] t = a2 - a1,
        // where t = [t1, t2].

        let rhs = &other.point - &self.point;
        let matrix = Matrix2::from_columns(&[self.dir, -other.dir]);

        // TODO: Rewrite to use LU decomposition?
        matrix
            .try_inverse()
            .map(|inv| inv * rhs)
            // Inverse returns vector, split it up into its components
            .map(|t| (t.x, t.y))
    }

    pub fn intersect_disk(&self, disk: &Disk<T>) -> Option<LineSegment2d<T>> {
        let [t1, t2] = self.intersect_disk_parametric(disk)?;
        let p1 = self.point_from_parameter(t1);
        let p2 = self.point_from_parameter(t2);
        Some(LineSegment2d::new(p1, p2))
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_disk_parametric(&self, disk: &Disk<T>) -> Option<[T; 2]> {
        let a = self.point();
        let d = self.dir();
        let r = disk.radius();
        let a0 = a - disk.center();

        // The solutions are given by the solutions to the quadratic equation
        // alpha * t^2 + beta * t + gamma = 0
        let alpha = d.dot(&d);
        let beta = 2.0 * d.dot(&a0);
        let gamma = a0.dot(&a0) - r * r;

        let discriminant = beta * beta - 4.0 * alpha * gamma;
        if discriminant >= 0.0 {
            // Non-negative discriminant means that we have two (possible identical) real solutions that correspond
            // to intersection points
            let disc_sqrt = discriminant.sqrt();
            let t1 = (-beta - disc_sqrt) / (2.0 * alpha);
            let t2 = (-beta + disc_sqrt) / (2.0 * alpha);
            debug_assert!(t1 <= t2);
            Some([t1, t2])
        } else {
            // No real solutions, so no intersection
            None
        }
    }
}
