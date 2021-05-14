use polyquad_parse::{parse2d, parse3d, ParseError};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::ffi::OsStr;
use std::fs::{read_dir, read_to_string};
use std::iter::once;
use std::path::{Path, PathBuf};
use std::{env, io};

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    generate_polyquad_rules::<2>(&out_dir, "tri", &Path::new("rules/polyquad/expanded/tri"));
    generate_polyquad_rules::<2>(&out_dir, "quad", &Path::new("rules/polyquad/expanded/quad"));
    generate_polyquad_rules::<3>(&out_dir, "tet", &Path::new("rules/polyquad/expanded/tet"));
    generate_polyquad_rules::<3>(&out_dir, "hex", &Path::new("rules/polyquad/expanded/hex"));
    generate_polyquad_rules::<3>(&out_dir, "pri", &Path::new("rules/polyquad/expanded/pri"));
    generate_polyquad_rules::<3>(&out_dir, "pyr", &Path::new("rules/polyquad/expanded/pyr"));

    println!("cargo:rerun-if-changed=rules/");
    println!("cargo:rerun-if-changed=build.rs");
}

#[derive(Debug)]
pub struct PolyquadRuleFile {
    /// Polynomial strength
    strength: usize,
    /// Number of quadrature points
    size: usize,
    path: PathBuf,
}

pub type QuadratureRule<const D: usize> = (Vec<f64>, Vec<[f64; D]>);

pub struct PolyquadRule<const D: usize> {
    strength: usize,
    weights: Vec<f64>,
    points: Vec<[f64; D]>,
}

/// Try to parse a filename like "2-5" into the respective strength (2) and quadrature size (5).
fn try_match_stem_to_strength_and_size(stem: &OsStr) -> Result<(usize, usize), ()> {
    let stem = stem.to_string_lossy();
    let mut iter = stem.split("-");

    let strength_str = iter.next().ok_or(())?;
    let size_str = iter.next().ok_or(())?;

    let strength: usize = strength_str.parse().map_err(|_| ())?;
    let size: usize = size_str.parse().map_err(|_| ())?;

    if iter.next().is_none() {
        Ok((strength, size))
    } else {
        Err(())
    }
}

fn find_polyquad_rule_files_in_dir(dir: impl AsRef<Path>) -> io::Result<Vec<PolyquadRuleFile>> {
    let dir = dir.as_ref();
    let mut rule_files = Vec::new();

    let iter = read_dir(dir)?;
    for entry in iter {
        let entry = entry?;
        if entry.metadata()?.is_file() {
            let filepath = entry.path();
            // valid filenames look like "5-20.txt",
            // where the first entry is the polynomial strength and the second is
            // the number of points

            match (filepath.extension(), filepath.file_stem()) {
                (Some(ext), Some(stem)) if ext == OsStr::new("txt") => {
                    if let Ok((strength, size)) = try_match_stem_to_strength_and_size(stem) {
                        rule_files.push(PolyquadRuleFile {
                            strength,
                            size,
                            path: filepath,
                        });
                    }
                }
                // Ignore the file if it does not fit the pattern
                _ => {}
            }
        }
    }

    Ok(rule_files)
}

struct Parser;

pub trait PolyquadParser<const D: usize> {
    fn parse(data: &str) -> Result<QuadratureRule<D>, ParseError>;
}

impl PolyquadParser<2> for Parser {
    fn parse(data: &str) -> Result<QuadratureRule<2>, ParseError> {
        parse2d(data).map(|rule| (rule.weights, rule.points))
    }
}

impl PolyquadParser<3> for Parser {
    fn parse(data: &str) -> Result<QuadratureRule<3>, ParseError> {
        parse3d(data).map(|rule| (rule.weights, rule.points))
    }
}

fn load_polyquad_rules<const D: usize>(rule_dir: &Path) -> Vec<PolyquadRule<D>>
where
    Parser: PolyquadParser<D>,
{
    let rule_files = find_polyquad_rule_files_in_dir(rule_dir).expect("Could not find rule files");
    let mut rules = Vec::new();
    for rule_file in rule_files {
        let data = read_to_string(&rule_file.path).expect(&format!(
            "Failed to load rule file {}",
            rule_file.path.display()
        ));
        let (weights, points) = Parser::parse(&data).expect(&format!(
            "Failed to parse polyquad rule file {}",
            rule_file.path.display()
        ));
        assert_eq!(
            rule_file.size,
            weights.len(),
            "Mismatch between expected size and actual size of quadrature rule"
        );
        assert_eq!(
            weights.len(),
            points.len(),
            "Mismatch between number of weights and points in quadrature rule."
        );
        rules.push(PolyquadRule {
            strength: rule_file.strength,
            weights,
            points,
        });
    }

    rules.sort_by_key(|rule| rule.strength);
    rules
}

fn generate_source_tokens_for_rules<const D: usize>(
    domain_name: &str,
    rules: Vec<PolyquadRule<D>>,
) -> TokenStream {
    let return_type = format_ident!("Rule{}d", D);

    let quadrature_tokens = rules.iter().map(|rule| {
        let strength = rule.strength;
        let fn_name = format_ident!("{}_{}", domain_name, strength);
        let weights = &rule.weights;
        let points_tokens = rule.points.iter().map(|point| {
            let x = point[0];
            let y = point[1];
            match point.len() {
                2 => quote!([#x, #y]),
                3 => {
                    let z = point[2];
                    quote!([#x, #y, #z])
                }
                _ => panic!("Unsupported dimension"),
            }
        });

        quote! {
            /// Auto-generated code.
            pub fn #fn_name() -> crate::#return_type {
                let weights = vec![#(#weights),*];
                let points = vec![#(#points_tokens),*];
                (weights, points)
            }
        }
    });

    let select_minimum_strength_tokens: TokenStream = {
        let max_strength = rules.iter().map(|rule| rule.strength).max().unwrap();
        let select_min_fn = format_ident!("{}_select_minimum", domain_name);

        let match_cases: TokenStream = rules
            .iter()
            .map(|rule| {
                let strength = rule.strength;
                let fn_name = format_ident!("{}_{}", domain_name, strength);
                quote! { #strength => Ok(#fn_name()), }
            })
            .collect();

        quote! {
            /// Auto-generated code
            pub fn #select_min_fn(strength: usize)
                -> Result<crate::#return_type, crate::polyquad::StrengthNotAvailable> {
                match strength {
                    #match_cases
                    s if s <= #max_strength => #select_min_fn(strength + 1),
                    _ => Err(crate::polyquad::StrengthNotAvailable)
                }
            }
        }
    };

    // Combine all tokens into a single TokenStream
    once(select_minimum_strength_tokens)
        .chain(quadrature_tokens)
        .collect()
}

fn write_tokens_to_file(tokens: &TokenStream, path: impl AsRef<Path>) {
    let code = format!("{:#}", tokens);
    let path = path.as_ref();
    std::fs::create_dir_all(path.parent().unwrap())
        .expect("Failed to create directory for generated output code");
    std::fs::write(&path, &code).expect("Failed to write source code for quadrature rule");
}

/// Generates code for the polyquad quadrature rules defined in the given directory and
/// with the given domain name (tri, tet etc.).
fn generate_polyquad_rules<const D: usize>(out_dir: &Path, domain_name: &str, rule_dir: &Path)
where
    Parser: PolyquadParser<D>,
{
    let rules = load_polyquad_rules::<D>(rule_dir.as_ref());
    let source_file = out_dir.join(&format!("polyquad/{}.rs", domain_name));
    let code_tokens = generate_source_tokens_for_rules(domain_name, rules);
    write_tokens_to_file(&code_tokens, &source_file);
    format_file(&source_file);
}

fn format_file(path: &Path) {
    let rustfmt_result = std::process::Command::new("rustfmt")
        .arg(format!("{}", path.display()))
        .output();
    if let Err(err) = rustfmt_result {
        eprintln!("Failed to run rustfmt on file {}: {}", path.display(), err);
        let warning = format!(
            "Failed to run rustfmt on generated file {}.\
                               Re-run with `-vv` for more output.",
            path.display()
        );
        println!("cargo:warning={}", warning);
    }
}
