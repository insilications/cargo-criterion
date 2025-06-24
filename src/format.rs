use std::fmt::Write;
use std::hash::{Hash, Hasher};

// A wrapper around f64 that can be used as a key in a HashMap/HashSet.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FloatKey(pub f64);

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

const POW10: [f64; 16] = [
    1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
#[inline]
fn pow10(exp: usize) -> f64 {
    POW10
        .get(exp)
        .copied()
        .unwrap_or_else(|| 10f64.powi(exp as i32))
}

pub struct PValueFormatter {
    short_precision: usize,
    long_precision: usize,
    p_value: f64,
    lower_bound: f64,
}

impl PValueFormatter {
    /// O(1) initialisation, performs all heavy math once.
    pub fn new(p_value: f64) -> Self {
        assert!(
            p_value.is_sign_positive() && p_value.is_finite(),
            "p-value must be positive and finite"
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let dec = (-p_value.log10()).ceil().max(0.0) as usize;
        let lb = p_value / pow10(dec);
        Self {
            short_precision: dec,
            long_precision: dec * 2,
            p_value,
            lower_bound: lb,
        }
    }

    /// Format a single `a`.
    /// O(d) where d is the number of output digits.
    pub fn fmt(&self, a: f64) -> String {
        let use_long = (a < self.p_value) & (a >= self.lower_bound);
        let prec = if use_long {
            self.long_precision
        } else {
            self.short_precision
        };
        let mut buf = String::with_capacity(32);
        write!(&mut buf, "{a:.prec$}").unwrap();
        buf
    }
}

pub fn change(pct: f64, signed: bool) -> String {
    if signed {
        format!("{:>+6}%", signed_short(pct * 1e2))
    } else {
        format!("{:>6}%", short(pct * 1e2))
    }
}

pub fn time(ns: f64) -> String {
    if ns < 1.0 {
        format!("{:>6} ps", short(ns * 1e3))
    } else if ns < 10f64.powi(3) {
        format!("{:>6} ns", short(ns))
    } else if ns < 10f64.powi(6) {
        format!("{:>6} us", short(ns / 1e3))
    } else if ns < 10f64.powi(9) {
        format!("{:>6} ms", short(ns / 1e6))
    } else {
        format!("{:>6} s", short(ns / 1e9))
    }
}

pub fn short(n: f64) -> String {
    if n < 10.0 {
        format!("{:.4}", n)
    } else if n < 100.0 {
        format!("{:.3}", n)
    } else if n < 1000.0 {
        format!("{:.2}", n)
    } else if n < 10000.0 {
        format!("{:.1}", n)
    } else {
        format!("{:.0}", n)
    }
}

fn signed_short(n: f64) -> String {
    let n_abs = n.abs();

    if n_abs < 10.0 {
        format!("{:+.4}", n)
    } else if n_abs < 100.0 {
        format!("{:+.3}", n)
    } else if n_abs < 1000.0 {
        format!("{:+.2}", n)
    } else if n_abs < 10000.0 {
        format!("{:+.1}", n)
    } else {
        format!("{:+.0}", n)
    }
}

pub fn iter_count(iterations: u64) -> String {
    if iterations < 10_000 {
        format!("{} iterations", iterations)
    } else if iterations < 1_000_000 {
        format!("{:.0}k iterations", (iterations as f64) / 1000.0)
    } else if iterations < 10_000_000 {
        format!("{:.1}M iterations", (iterations as f64) / (1000.0 * 1000.0))
    } else if iterations < 1_000_000_000 {
        format!("{:.0}M iterations", (iterations as f64) / (1000.0 * 1000.0))
    } else if iterations < 10_000_000_000 {
        format!(
            "{:.1}B iterations",
            (iterations as f64) / (1000.0 * 1000.0 * 1000.0)
        )
    } else {
        format!(
            "{:.0}B iterations",
            (iterations as f64) / (1000.0 * 1000.0 * 1000.0)
        )
    }
}

pub fn integer(n: f64) -> String {
    format!("{}", n as u64)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn short_max_len() {
        let mut float = 1.0;
        while float < 999_999.9 {
            let string = short(float);
            println!("{}", string);
            assert!(string.len() <= 6);
            float *= 2.0;
        }
    }

    #[test]
    fn signed_short_max_len() {
        let mut float = -1.0;
        while float > -999_999.9 {
            let string = signed_short(float);
            println!("{}", string);
            assert!(string.len() <= 7);
            float *= 2.0;
        }
    }

    /// For readability we store examples and expected strings side-by-side.
    fn check_cases(p_val: f64, cases: &[(f64, &str)]) {
        let fmt = PValueFormatter::new(p_val);
        for &(a, expected) in cases {
            assert_eq!(
                fmt.fmt(a),
                expected,
                "p_value={p_val}, a={a} should format to `{expected}`"
            );
        }
    }

    // ---------- canonical examples -------------------------------------
    #[test]
    fn matches_original_example_output() {
        /* Examples copied from the original snippet
           --------------------------------------------------------------
           Expected strings were produced by the reference programme
           and hand-verified.
        */
        const CASES: &[(f64, &str)] = &[
            (0.000_000_1, "0.000"),
            (0.000_000_123, "0.000"),
            (0.000_001_23, "0.000001"),
            (0.000_002_23, "0.000002"),
            (0.000_001, "0.000001"),
            (0.000_000_23, "0.000"),
            (0.000_012_3, "0.000012"),
            (0.4723, "0.472"),
            (0.000_001, "0.000001"),
        ];
        check_cases(0.001, CASES);
    }

    // ---------- boundary conditions ------------------------------------
    #[test]
    fn lower_bound_gets_long_precision() {
        // lower_bound = p / 10^decimals
        let p = 0.001_f64;
        let lb = 0.000_001_f64; // manually computed
        let fmt = PValueFormatter::new(p);

        // long precision is 6 ⇒ we expect 6 decimals
        assert_eq!(fmt.fmt(lb), "0.000001");
    }

    #[test]
    fn equal_to_p_value_gets_short_precision() {
        let p = 0.001_f64;
        let fmt = PValueFormatter::new(p);

        // short precision is 3 ⇒ "0.001"
        assert_eq!(fmt.fmt(p), "0.001");
    }

    #[test]
    fn above_p_value_gets_short_precision() {
        let p = 0.001_f64;
        let fmt = PValueFormatter::new(p);

        // short precision = 3
        assert_eq!(fmt.fmt(0.10), "0.100");
    }

    // ---------- negative / unusual 'a' --------------------------------
    #[test]
    fn negative_a_is_supported() {
        let p = 0.001_f64;
        let fmt = PValueFormatter::new(p);

        // a < lower_bound ⇒ short precision (3)
        assert_eq!(fmt.fmt(-0.0005), "-0.001");
    }

    // ---------- constructor validity checks ---------------------------
    #[test]
    #[should_panic(expected = "p-value must be positive")]
    fn new_panics_on_non_positive_p() {
        let _ = PValueFormatter::new(0.0);
    }

    #[test]
    #[should_panic(expected = "p-value must be positive")]
    fn new_panics_on_negative_p() {
        let _ = PValueFormatter::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "p-value must be positive")]
    fn new_panics_on_nan() {
        let _ = PValueFormatter::new(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "p-value must be positive")]
    fn new_panics_on_infinite() {
        let _ = PValueFormatter::new(f64::INFINITY);
    }

    // ---------- repeated-use correctness & amortisation ---------------
    #[test]
    fn repeated_calls_yield_identical_results() {
        let fmt = PValueFormatter::new(1e-5);
        let inputs = [2.3e-7, 5.0e-6, 0.15, 2.3e-7];
        // Capture once
        let first_pass: Vec<String> = inputs.iter().map(|&a| fmt.fmt(a)).collect();
        // Second pass must match
        let second_pass: Vec<String> = inputs.iter().map(|&a| fmt.fmt(a)).collect();
        assert_eq!(first_pass, second_pass);
    }
}
