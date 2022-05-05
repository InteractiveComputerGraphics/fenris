use crate::{ConvexPolygon, Disk, HalfPlane, Plane};
use nalgebra::{clamp, DefaultAllocator, DimName, Matrix2, OPoint, OVector, RealField, U2, U3, Vector2};
use nalgebra::{Point2, Point3, Scalar};
use numeric_literals::replace_float_literals;
use std::fmt::Debug;
use nalgebra::allocator::Allocator;

pub type LineSegment3d<T> = LineSegment<T, U3>;

impl<T: RealField> LineSegment3d<T> {
    #[allow(non_snake_case)]
    pub fn closest_point_to_plane_parametric(&self, plane: &Plane<T>) -> T {
        let n = plane.normal();
        let x0 = plane.point();
        let [a, b] = [self.start(), self.end()];
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

    pub fn closest_point_to_plane(&self, plane: &Plane<T>) -> Point3<T> {
        let t = self.closest_point_to_plane_parametric(plane);
        self.point_from_parameter(t)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineSegment<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    start: OPoint<T, D>,
    end: OPoint<T, D>,
}

pub type LineSegment2d<T> = LineSegment<T, U2>;

impl<T, D> LineSegment<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn from_end_points(start: OPoint<T, D>, end: OPoint<T, D>) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> &OPoint<T, D> {
        &self.start
    }

    pub fn end(&self) -> &OPoint<T, D> {
        &self.end
    }

    pub fn reverse(&self) -> Self {
        Self {
            start: self.end.clone(),
            end: self.start.clone(),
        }
    }
}

impl<T, D> LineSegment<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn to_line(&self) -> Line<T, D> {
        let dir = &self.end - &self.start;
        Line::from_point_and_dir(self.start.clone(), dir)
    }

    /// Returns the vector tangent to the line, pointing from start to end.
    ///
    /// Note that the vector is **not** normalized.
    pub fn tangent_dir(&self) -> OVector<T, D> {
        &self.end().coords - &self.start().coords
    }

    pub fn length(&self) -> T {
        self.tangent_dir().norm()
    }

    pub fn midpoint(&self) -> OPoint<T, D> {
        OPoint::from((&self.start().coords + &self.end().coords) / (T::one() + T::one()))
    }

    /// Compute the closest point on the segment to the given point, represented in
    /// the parametric form x = a + t * (b - a).
    pub fn closest_point_parametric(&self, point: &OPoint<T, D>) -> T {
        let t = self.to_line().project_point_parametric(point);
        clamp(t, T::zero(), T::one())
    }

    /// Computes the closest point on the line to the given point.
    pub fn closest_point(&self, point: &OPoint<T, D>) -> OPoint<T, D> {
        let t = self.closest_point_parametric(point);
        self.point_from_parameter(t)
    }

    pub fn point_from_parameter(&self, t: T) -> OPoint<T, D> {
        OPoint::from(&self.start().coords + self.tangent_dir() * t)
    }
}

impl<T> LineSegment2d<T>
where
    T: RealField,
{
    /// Returns a vector normal to the line segment, in the direction consistent with a
    /// counter-clockwise winding order when the edge is part of a polygon.
    ///
    /// Note that the vector is **not** normalized.
    pub fn normal_dir(&self) -> Vector2<T> {
        let tangent = self.tangent_dir();
        Vector2::new(tangent.y, -tangent.x)
    }

    pub fn intersect_line_parametric(&self, line: &Line2d<T>) -> Option<T> {
        self.to_line()
            .intersect_line_parametric(line)
            .map(|(t1, _)| t1)
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_disk_parametric(&self, disk: &Disk<T>) -> Option<[T; 2]> {
        let [t1, t2] = self.to_line().intersect_disk_parametric(disk)?;
        let t1 = clamp(t1, 0.0, 1.0);
        let t2 = clamp(t2, 0.0, 1.0);
        Some([t1, t2])
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_disk(&self, disk: &Disk<T>) -> Option<Self> {
        self.intersect_disk_parametric(disk)
            .map(|[t1, t2]| self.segment_from_parameters(&t1, &t2))
    }

    pub fn segment_from_parameters(&self, t_begin: &T, t_end: &T) -> Self {
        let begin = self.point_from_parameter(t_begin.clone());
        let end = self.point_from_parameter(t_end.clone());
        Self::from_end_points(begin, end)
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

impl<T: RealField> LineSegment3d<T> {
    pub fn intersect_plane_parametric(&self, plane: &Plane<T>) -> Option<T> {
        self.to_line()
            .intersect_plane_parametric(plane)
            .filter(|t| t >= &T::zero() && t <= &T::one())
    }
}

#[derive(Debug, Clone)]
pub struct Line<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    point: OPoint<T, D>,
    dir: OVector<T, D>,
}

pub type Line2d<T> = Line<T, U2>;
pub type Line3d<T> = Line<T, U3>;

impl<T, D> Line<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn from_point_and_dir(point: OPoint<T, D>, dir: OVector<T, D>) -> Self {
        // TODO: Make dir Unit?
        Self { point, dir }
    }

    pub fn point(&self) -> &OPoint<T, D> {
        &self.point
    }

    pub fn dir(&self) -> &OVector<T, D> {
        &self.dir
    }
}

impl<T, D> Line<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    /// A normalized vector tangent to the line.
    pub fn tangent(&self) -> OVector<T, D> {
        self.dir.normalize()
    }

    pub fn from_point_through_point(point: OPoint<T, D>, through: &OPoint<T, D>) -> Self {
        let dir = through - &point;
        Self::from_point_and_dir(point, dir)
    }

    /// Computes the projection of the given point onto the line, representing the point
    /// in parametric form.
    pub fn project_point_parametric(&self, point: &OPoint<T, D>) -> T {
        let d2 = self.dir.magnitude_squared();
        if d2 == T::zero() {
            // TODO: Is this the correct way to handle it? If the line degenerate to a point
            // it's no longer even a line! (It is a line *segment* though ...)
            // Line degenerates to a point, just return 0 as it will give *a* correct solution
            T::zero()
        } else {
            (point - &self.point).dot(&self.dir) / d2
        }
    }

    /// Computes the projection of the given point onto the line.
    pub fn project_point(&self, point: &OPoint<T, D>) -> OPoint<T, D> {
        let t = self.project_point_parametric(point);
        self.point_from_parameter(t)
    }

    pub fn point_from_parameter(&self, t: T) -> OPoint<T, D> {
        &self.point + &self.dir * t
    }
}

impl<T> Line2d<T>
where
    T: RealField,
{
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
        Some(LineSegment2d::from_end_points(p1, p2))
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

impl<T> Line3d<T>
where
    T: RealField,
{
    pub fn intersect_plane_parametric(&self, plane: &Plane<T>) -> Option<T> {
        let n = plane.normal();
        let d = self.dir();
        let b = self.point() - plane.point();
        let d_dot_n = d.dot(&n);
        // TODO: This will actually *never* return a non-empty intersection in the case
        // that the line is entirely contained in the plane. However, this is so extremely
        // unlikely to be the case in the presence of floating-point arithmetic, that we
        // consider it never to be the case
        (d_dot_n != T::zero())
            .then(|| - b.dot(&n) / d_dot_n)
    }
}