use polyquad_parse::{parse2d, Rule2d};
use std::path::{PathBuf, Path};
use std::fs::{read_dir, read_to_string};
use std::{io, env};
use std::ffi::OsStr;
use std::iter::once;
use quote::{format_ident, quote};
use proc_macro2::TokenStream;

#[derive(Debug)]
pub struct PolyquadRuleFile {
    /// Polynomial strength
    strength: usize,
    /// Number of quadrature points
    size: usize,
    path: PathBuf
}

pub struct PolyquadRule2d {
    strength: usize,
    rule: Rule2d
}

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
                            path: filepath
                        });
                    }
                },
                // Ignore the file if it does not fit the pattern
                _ => {}
            }

        }
    }

    Ok(rule_files)
}

fn generate_polyquad_rules(out_dir: &Path, rule_dir: impl AsRef<Path>, domain_name: &str) {
    let rule_files = find_polyquad_rule_files_in_dir(rule_dir)
        .expect("Could not find rule files");
    let mut rules = Vec::new();
    for rule_file in rule_files {
        let data = read_to_string(&rule_file.path)
            .expect(&format!("Failed to load rule file {}", rule_file.path.display()));
        let rule = parse2d(&data)
            .expect(&format!("Failed to parse polyquad rule file {}", rule_file.path.display()));
        assert_eq!(rule_file.size, rule.weights.len(),
                   "Mismatch between expected size and actual size of quadrature rule");
        assert_eq!(rule.weights.len(), rule.points.len(),
                   "Mismatch between number of weights and points in quadrature rule.");
        rules.push(PolyquadRule2d {
            strength: rule_file.strength,
            rule
        });
    }

    rules.sort_by_key(|rule| rule.strength);

    let source_file = out_dir.join(&format!("polyquad/{}.rs", domain_name));

    let quadrature_tokens = rules
        .iter()
        .map(|rule| {
            let strength = rule.strength;
            let fn_name = format_ident!("{}_{}", domain_name, strength);
            let weights = &rule.rule.weights;
            let points_tokens = rule.rule.points
                .iter()
                .map(|&[x, y]| quote!([#x, #y]));

            let tokens: TokenStream = quote! {
                /// Auto-generated code.
                pub fn #fn_name() -> crate::Rule2d {
                    let weights = vec![#(#weights),*];
                    let points = vec![#(#points_tokens),*];
                    (weights, points)
                }
            };
            tokens
        });

    let match_cases: TokenStream = rules
        .iter()
        .map(|rule| {
            let strength = rule.strength;
            let fn_name = format_ident!("{}_{}", domain_name, strength);
            quote! { #strength => Ok(#fn_name()), }
        }).collect();

    let select_fn = format_ident!("{}_select_exact", domain_name);
    let select_exact_tokens: TokenStream = quote! {
        pub fn #select_fn(strength: usize)
            -> Result<crate::Rule2d, crate::polyquad::StrengthNotAvailable> {
            match strength {
                #match_cases
                _ => Err(crate::polyquad::StrengthNotAvailable)
            }
        }
    };

    let select_minimum_strength_tokens: TokenStream = {
        let max_strength = rules.iter().map(|rule| rule.strength).max().unwrap();
        let select_min_fn = format_ident!("{}_select_minimum", domain_name);
        quote! {
            /// Auto-generated code
            pub fn #select_min_fn(strength: usize)
                -> Result<crate::Rule2d, crate::polyquad::StrengthNotAvailable> {
                match strength {
                    #match_cases
                    s if s <= #max_strength => #select_min_fn(strength + 1),
                    _ => Err(crate::polyquad::StrengthNotAvailable)
                }
            }
        }
    };

    let code_tokens: TokenStream = once(select_minimum_strength_tokens)
        .chain(once(select_exact_tokens))
        .chain(quadrature_tokens)
        .collect();

    let code = format!("{:#}", code_tokens);
    std::fs::create_dir_all(source_file.parent().unwrap())
        .expect("Failed to create directory for generated output code");
    std::fs::write(&source_file, &code)
        .expect("Failed to write source code for quadrature rule");

    format_file(&source_file);
}

fn format_file(path: &Path) {
    let rustfmt_result = std::process::Command::new("rustfmt")
        .arg(format!("{}", path.display()))
        .output();
    if let Err(err) = rustfmt_result {
        eprintln!("Failed to run rustfmt on file {}: {}", path.display(), err);
        let warning = format!("Failed to run rustfmt on generated file {}.\
                               Re-run with `-vv` for more output.",
                              path.display());
        println!("cargo:warning={}", warning);
    }
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    generate_polyquad_rules(&out_dir, "rules/polyquad/expanded/tri", "tri");
    generate_polyquad_rules(&out_dir, "rules/polyquad/expanded/quad", "quad");

    println!("cargo:rerun-if-changed=rules/");
    println!("cargo:rerun-if-changed=build.rs");
}