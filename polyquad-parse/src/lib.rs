//! Tiny parser for the output of `polyquad` quadrature files.
//!
//! This crate provides parsers for 2D and 3D quadrature rules formatted in a simple text file
//! format and bundled with the paper
//!
//! ```text
//! Witherden, Freddie D., and Peter E. Vincent.
//! "On the identification of symmetric quadrature rules for finite element methods."
//! Computers & Mathematics with Applications 69, no. 10 (2015): 1232-1241.
//! ```
//!
//! It is used as a (build) dependency of `fenris-quadrature`.
//!
//! It is only minimally tested, because the resulting quadrature rules are further tested in
//! `fenris-quadrature`.

use core::fmt;
use core::fmt::{Display, Formatter};
use std::error::Error;

pub type Point2 = [f64; 2];
pub type Point3 = [f64; 3];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    error: String,
}

impl ParseError {
    fn from_string(error: String) -> Self {
        Self { error }
    }

    fn float_parse_error(str: &str, label: &str, error: impl Error) -> Self {
        ParseError::from_string(format!(
            "Failed to parse {} ({}) as f64: {}",
            str, label, error
        ))
    }

    fn unexpected_entries(num_entries_found: usize, num_entries_expected: usize) -> Self {
        ParseError::from_string(format!(
            "Found {} entries in line, but expected {}",
            num_entries_found, num_entries_expected
        ))
    }
}

fn try_parse_f64(str: &str, label: &str) -> Result<f64, ParseError> {
    str.parse::<f64>()
        .map_err(|err| ParseError::float_parse_error(str, label, err))
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Default, PartialEq)]
pub struct Rule2d {
    pub points: Vec<Point2>,
    pub weights: Vec<f64>,
}

#[derive(Clone, Default, PartialEq)]
pub struct Rule3d {
    pub points: Vec<Point3>,
    pub weights: Vec<f64>,
}

fn parse_helper(
    data: &str,
    labels: &[&str],
    mut handler: impl FnMut(&[f64]) -> Result<(), ParseError>,
) -> Result<(), ParseError> {
    let mut line_numbers = Vec::new();

    for line in data.lines() {
        let mut iter = line.split_ascii_whitespace().peekable();

        // Skip empty lines
        if iter.peek().is_some() {
            line_numbers.clear();

            for (i, entry) in iter.enumerate() {
                let label = labels.get(i).unwrap_or_else(|| &"unlabeled entry");
                let coord_entry = try_parse_f64(entry, label)?;
                line_numbers.push(coord_entry);
            }

            handler(&line_numbers)?;
        }
    }
    Ok(())
}

/// Attempts to parse a text file in the `expanded` format as a 2D quadrature rule.
///
/// The text file should contain lines with the format
/// ```text
///    <x>     <y>     <weight>
/// ```
/// where `x`, `y` and `weights` are floating-point numbers.
pub fn parse2d(data: &str) -> Result<Rule2d, ParseError> {
    let mut rule = Rule2d::default();
    parse_helper(
        data,
        &["x coordinate", "y coordinate", "weight"],
        |entries| match entries {
            [x, y, w] => {
                rule.points.push([*x, *y]);
                rule.weights.push(*w);
                Ok(())
            }
            _ => Err(ParseError::unexpected_entries(entries.len(), 3)),
        },
    )?;
    Ok(rule)
}

/// Attempts to parse a text file in the `expanded` format as a 3D quadrature rule.
///
/// The text file should contain lines with the format
/// ```text
///    <x>     <y>     <z>     <weight>
/// ```
/// where `x`, `y`, `z` and `weights` are floating-point numbers delimited by whitespace.
pub fn parse3d(data: &str) -> Result<Rule3d, ParseError> {
    let mut rule = Rule3d::default();
    let labels = &["x coordinate", "y coordinate", "z coordinate", "weight"];
    parse_helper(data, labels, |entries| match entries {
        [x, y, z, w] => {
            rule.points.push([*x, *y, *z]);
            rule.weights.push(*w);
            Ok(())
        }
        _ => Err(ParseError::unexpected_entries(entries.len(), 4)),
    })?;
    Ok(rule)
}
