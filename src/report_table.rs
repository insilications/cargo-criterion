use crate::format;
use crate::model::{Benchmark, BenchmarkGroup, Model};
use crate::report::{
    compare_to_threshold, BenchmarkId, ComparisonData, ComparisonResult, OwnedMeasurementData,
};
use crate::value_formatter::ValueFormatter;
use crate::{
    estimate::{ConfidenceInterval, Estimates},
    model::SavedStatistics,
};
// use crate::report::{
//     compare_to_threshold, rank_fastest_with_scores, BenchmarkId, ComparisonReport,
//     ComparisonReportRanking, ComparisonReportRankingData, ComparisonReportRankingResult,
//     ComparisonResult, OwnedMeasurementData,
// };
use itertools::Itertools;
use tabled::{
    grid::config::ColoredConfig,
    grid::records::{ExactRecords, PeekableRecords, Records},
    settings::{style::Style, themes::BorderCorrection, Alignment, Format, TableOption},
    Table, Tabled,
};

use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::{collections::hash_map::Entry, ops::Range};

pub struct ComparisonReport<'benchmark_group> {
    pub id_new: &'benchmark_group BenchmarkId,
    pub id_old: &'benchmark_group BenchmarkId,
    pub benchmark_new: &'benchmark_group Benchmark,
    pub benchmark_old: &'benchmark_group Benchmark,
    pub comp: ComparisonData,
    pub ranking_result: ComparisonReportRankingResult,
}

impl<'benchmark_group> ComparisonReport<'benchmark_group> {
    pub fn new(
        id_new: &'benchmark_group BenchmarkId,
        id_old: &'benchmark_group BenchmarkId,
        benchmark_new: &'benchmark_group Benchmark,
        benchmark_old: &'benchmark_group Benchmark,
        comp: ComparisonData,
    ) -> Self {
        Self {
            id_new,
            id_old,
            benchmark_new,
            benchmark_old,
            comp,
            ranking_result: ComparisonReportRankingResult::NoChange,
        }
    }
}

#[derive(Debug)]
pub enum ComparisonReportRankingResult {
    Improved,                // +2 score for new, -2 score for old
    Regressed,               // +2 score for old, -2 score for new
    NonSignificantImproved,  // +1 score for new, -1 score for old
    NonSignificantRegressed, // +1 score for old, -1 score for new
    NoChange,                //  0 score
}

#[derive(Debug)]
pub struct ComparisonReportRanking {
    pub function_id_new: String,
    pub function_id_old: String,
    pub result: ComparisonReportRankingResult,
}

pub struct ComparisonReportRankingData {
    pub latency_mean_str: String,
    pub latency_mean: f64,
    pub latency_mean_ci: ConfidenceInterval,
}

/// Fast-to-slow ranking plus scores.
#[derive(Debug)]
pub struct RankingResult {
    /// Vector of ranking tiers (fastest → slowest).
    /// Each inner `Vec<String>` holds all function IDs that are tied.
    pub ranks: Vec<Vec<String>>,

    /// Score – identical for every member of the same tier. For debugging purposes.
    pub scores: HashMap<String, i32>,
}

/// Score-based ranking WITHOUT equivalence classes.
///
/// • Every function keeps its own score; `NoChange` contributes 0.
/// • Functions are ranked by that score (descending).
/// • If two or more functions end up with the *exact* same score
///   they share a tier, because the score alone cannot order them.
///
/// Complexity
///   n = #reports, m = #distinct function IDs
///   • scoring loop            : O(n)
///   • sort by score           : O(m log m)  (dominant)
///   • total memory            : O(m)
///
/// All data structures are pre-allocated where possible; the function
/// uses only safe Rust and does not perform unnecessary cloning.
pub fn rank_fastest_with_scores<'benchmark_group>(
    comparisons_report: &'benchmark_group [ComparisonReport<'benchmark_group>],
) -> RankingResult {
    // 0. Map every unique function ID → dense index 0‥m-1
    let mut id_to_idx: HashMap<String, usize> =
        HashMap::with_capacity(comparisons_report.len() * 2); // rough upper bound
    let mut next_idx = 0usize;

    for r in comparisons_report {
        // new side
        id_to_idx
            .entry(r.id_new.function_id.as_ref().unwrap().clone())
            .or_insert_with(|| {
                let idx = next_idx;
                next_idx += 1;
                idx
            });
        // old side
        id_to_idx
            .entry(r.id_old.function_id.as_ref().unwrap().clone())
            .or_insert_with(|| {
                let idx = next_idx;
                next_idx += 1;
                idx
            });
    }

    // 1. Per-ID score vector
    let mut score = vec![0i32; next_idx];

    for r in comparisons_report {
        let a = id_to_idx[r.id_new.function_id.as_ref().unwrap()];
        let b = id_to_idx[r.id_old.function_id.as_ref().unwrap()];
        match r.ranking_result {
            ComparisonReportRankingResult::Improved => {
                // +2 score for new, -2 score for old
                score[a] += 2;
                score[b] -= 2;
            }
            ComparisonReportRankingResult::Regressed => {
                // +2 score for old, -2 score for new
                score[a] -= 2;
                score[b] += 2;
            }
            ComparisonReportRankingResult::NonSignificantImproved => {
                // +1 score for new, -1 score for old
                score[a] += 1;
                score[b] -= 1;
            }
            ComparisonReportRankingResult::NonSignificantRegressed => {
                // +1 score for old, -1 score for new
                score[a] -= 1;
                score[b] += 1;
            }
            ComparisonReportRankingResult::NoChange => { /* 0 pts */ }
        }
    }

    // 2. Build score maps
    let mut id_to_score: HashMap<String, i32> = HashMap::with_capacity(id_to_idx.len());
    for (id, &idx) in &id_to_idx {
        id_to_score.insert(id.clone(), score[idx]);
    }

    // 3. Sort by score (descending) and group ties
    let mut entries: Vec<(String, i32)> =
        id_to_score.iter().map(|(id, &s)| (id.clone(), s)).collect();

    entries.sort_unstable_by(|a, b| {
        let ord = b.1.cmp(&a.1); // score DESC
        if ord == std::cmp::Ordering::Equal {
            a.0.cmp(&b.0) // name ASC (stable tie-break)
        } else {
            ord
        }
    });

    let mut ranks: Vec<Vec<String>> = Vec::with_capacity(12);
    for (id, s) in entries {
        if ranks
            .last()
            .is_none_or(|g: &Vec<String>| id_to_score[&g[0]] != s)
        {
            ranks.push(vec![id]); // new tier
        } else {
            ranks.last_mut().unwrap().push(id); // same tier
        }
    }

    // ──────────────────────────────────────────────────────────────
    RankingResult {
        ranks,
        scores: id_to_score,
    }
}

pub struct GroupsComparisons(HashMap<String, GroupComparisonTables>);

impl GroupsComparisons {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }
}

impl Deref for GroupsComparisons {
    type Target = HashMap<String, GroupComparisonTables>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GroupsComparisons {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl fmt::Display for GroupsComparisons {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (group_id, comparison_tables) in &**self {
            let mut changes_table = Table::new(&comparison_tables.changes_table_rows);
            // Changes `ChangesTable::function_id_vs` column name to `group_id`
            changes_table.modify((0, 0), Format::content(|_| group_id.to_string()));
            changes_table
                .with(Style::modern())
                .with(Alignment::center())
                .with(Alignment::center_vertical());
            writeln!(f, "{changes_table}")?;

            let mut ranking_table = Table::new(&comparison_tables.ranking_table_rows);
            ranking_table
                .with(Style::modern())
                .with(MergeDuplicatesVerticalFirst)
                .with(BorderCorrection::span())
                .with(Alignment::center())
                .with(Alignment::center_vertical());
            writeln!(f, "{ranking_table}")?;
        }
        Ok(())
    }
}

pub struct GroupComparisonTables {
    changes_table_rows: Vec<ChangesTable>,
    ranking_table_rows: Vec<RankingTable>,
}

pub struct IntraGroupComparison {
    comparison_tables: GroupsComparisons,
}

struct Bold<T>(T);

impl<T: fmt::Display> fmt::Display for Bold<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[1m{}\x1B[22m", self.0)
    }
}

#[inline]
fn bold<T: fmt::Display>(s: T) -> Bold<T> {
    Bold(s)
}

struct Green<T>(T);

impl<T: fmt::Display> fmt::Display for Green<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[32m{}\x1B[39m", self.0)
    }
}

#[inline]
fn green<T: fmt::Display>(s: T) -> Green<T> {
    Green(s)
}

struct Red<T>(T);

impl<T: fmt::Display> fmt::Display for Red<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[31m{}\x1B[39m", self.0)
    }
}

#[inline]
fn red<T: fmt::Display>(s: T) -> Red<T> {
    Red(s)
}

struct Faint<T>(T);

impl<T: fmt::Display> fmt::Display for Faint<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[2m{}\x1B[22m", self.0)
    }
}

#[inline]
fn faint<T: fmt::Display>(s: T) -> Faint<T> {
    Faint(s)
}

impl IntraGroupComparison {
    pub fn new() -> Self {
        Self {
            comparison_tables: GroupsComparisons::with_capacity(12),
        }
    }

    pub fn get_intra_group_comparison_data<'group_id, 'formatter, 'benchmark_group>(
        &mut self,
        group_id: &'group_id str,
        benchmark_group: &'benchmark_group BenchmarkGroup,
        formatter: &'formatter ValueFormatter,
    ) {
        let mut comparisons_report: Vec<ComparisonReport<'benchmark_group>> =
            Vec::with_capacity(12);

        for combinations in benchmark_group.benchmarks.iter().tuple_combinations::<(
            (&'benchmark_group BenchmarkId, &'benchmark_group Benchmark),
            (&'benchmark_group BenchmarkId, &'benchmark_group Benchmark),
        )>() {
            let ((id_new, benchmark_new), (id_old, benchmark_old)): (
                (&'benchmark_group BenchmarkId, &'benchmark_group Benchmark),
                (&'benchmark_group BenchmarkId, &'benchmark_group Benchmark),
            ) = combinations;

            let comp: ComparisonData = crate::analysis::analysis_comparison(
                                        benchmark_new.config.as_ref().unwrap(),
                                        &benchmark_new
                                            .raw_analysis_results
                                            .as_ref()
                                            .map(|r: &OwnedMeasurementData| -> crate::analysis::MeasuredValues<'_> {
                                                crate::analysis::MeasuredValues {
                                                    iteration_count: &r.iter_counts,
                                                    sample_values: &r.sample_times,
                                                    avg_values: &r.avg_times,
                                                }
                                            })
                                            .unwrap(),
                                        &benchmark_old
                                            .raw_analysis_results
                                            .as_ref()
                                            .map(
                                                |r: &OwnedMeasurementData| -> (
                                                    crate::analysis::MeasuredValues<'_>,
                                                    &'_ Estimates,
                                                ) {
                                                    (
                                                        crate::analysis::MeasuredValues {
                                                            iteration_count: &r.iter_counts,
                                                            sample_values: &r.sample_times,
                                                            avg_values: &r.avg_times,
                                                        },
                                                        &r.absolute_estimates,
                                                    )
                                                },
                                            )
                                            .unwrap(),
                                    );
            comparisons_report.push(ComparisonReport::<'benchmark_group>::new(
                id_new,
                id_old,
                benchmark_new,
                benchmark_old,
                comp,
            ));
            // comparisons_report.push(ComparisonReport::<'benchmark_group> {
            //     id_new,
            //     id_old,
            //     benchmark_new,
            //     benchmark_old,
            //     comp,
            // });
        }

        if !comparisons_report.is_empty() {
            if let Some(entry) = self.comparison_tables.insert(
                group_id.to_owned(),
                Self::parse_comparisons(&mut comparisons_report, formatter),
            ) {
                eprintln!("ALREADY INSERTED: {group_id}");
            } else {
                eprintln!("NOT INSERTED: {group_id}");
            }
        }
    }

    fn parse_comparisons<'benchmark_group>(
        my_comparisons_report: &'benchmark_group mut Vec<ComparisonReport<'benchmark_group>>,
        formatter: &ValueFormatter,
    ) -> GroupComparisonTables {
        // let mut comparison_report_results: Vec<ComparisonReportRanking> = Vec::with_capacity(12);
        let mut p_value_formatters: HashMap<format::FloatKey, format::PValueFormatter> =
            HashMap::with_capacity(12);
        let mut changes_table_rows: Vec<ChangesTable> = Vec::with_capacity(12);

        let mut functions_comparison_report_data: HashMap<String, ComparisonReportRankingData> =
            HashMap::with_capacity(12);

        for comparison in my_comparisons_report.iter_mut() {
            let comp = &comparison.comp;
            let significance_threshold = comp.significance_threshold;
            let is_mean_different = comp.p_value < significance_threshold;
            let mean_diff_est = &comp.relative_estimates.mean;
            let mean_diff_point_estimate = mean_diff_est.point_estimate;

            let benchmark_old_mean = comparison
                .benchmark_old
                .raw_analysis_results
                .as_ref()
                .unwrap()
                .absolute_estimates
                .mean
                .point_estimate;
            let benchmark_new_mean = comparison
                .benchmark_new
                .raw_analysis_results
                .as_ref()
                .unwrap()
                .absolute_estimates
                .mean
                .point_estimate;

            let benchmark_old_mean_ci = comparison
                .benchmark_old
                .raw_analysis_results
                .as_ref()
                .unwrap()
                .absolute_estimates
                .mean
                .confidence_interval
                .clone();

            let benchmark_new_mean_ci = comparison
                .benchmark_new
                .raw_analysis_results
                .as_ref()
                .unwrap()
                .absolute_estimates
                .mean
                .confidence_interval
                .clone();

            let mean_diff_ci = &mean_diff_est.confidence_interval;
            let mean_diff_ci_lower_bound = mean_diff_ci.lower_bound * benchmark_old_mean;
            let mean_diff_ci_upper_bound = mean_diff_ci.upper_bound * benchmark_old_mean;
            let mean_diff_pct_str = format!("{:.2}%", mean_diff_point_estimate.abs() * 1e2);
            let noise_threshold = comp.noise_threshold;
            let function_id_old_str = comparison.id_old.function_id.as_ref().unwrap().to_owned();
            let function_id_new_str = comparison.id_new.function_id.as_ref().unwrap().to_owned();
            let explanation_str: String;

            let p_value_formatter = p_value_formatters
                .entry(format::FloatKey(comp.p_value))
                .or_insert_with(|| format::PValueFormatter::new(significance_threshold));
            let mut mean_diff = format!("{:+.2} ns", mean_diff_point_estimate * benchmark_old_mean);
            let mut function_id_old_color_str = function_id_old_str.clone();
            let mut function_id_new_color_str = function_id_new_str.clone();
            let mut benchmark_old_mean_str = formatter.format_value(benchmark_old_mean);
            let mut benchmark_new_mean_str = formatter.format_value(benchmark_new_mean);
            functions_comparison_report_data.insert(
                function_id_new_str.clone(),
                ComparisonReportRankingData {
                    latency_mean_str: benchmark_new_mean_str.clone(),
                    latency_mean: benchmark_new_mean,
                    latency_mean_ci: benchmark_new_mean_ci,
                },
            );
            functions_comparison_report_data.insert(
                function_id_old_str.clone(),
                ComparisonReportRankingData {
                    latency_mean_str: benchmark_old_mean_str.clone(),
                    latency_mean: benchmark_old_mean,
                    latency_mean_ci: benchmark_old_mean_ci,
                },
            );

            if is_mean_different {
                let comparison_result = compare_to_threshold(mean_diff_est, noise_threshold);
                match comparison_result {
                    ComparisonResult::Improved => {
                        mean_diff = green(bold(mean_diff)).to_string();
                        benchmark_new_mean_str = green(bold(benchmark_new_mean_str)).to_string();
                        benchmark_old_mean_str = red(benchmark_old_mean_str).to_string();
                        function_id_new_color_str =
                            green(bold(function_id_new_color_str)).to_string();
                        function_id_old_color_str = red(function_id_old_color_str).to_string();
                        explanation_str = format!(
                            "Performance has {}",
                            green(bold(format!("improved {mean_diff_pct_str}")))
                        );
                        // comparison_report_results.push(ComparisonReportRanking {
                        //     function_id_new: function_id_new_str,
                        //     function_id_old: function_id_old_str,
                        //     result: ComparisonReportRankingResult::Improved,
                        // });
                        comparison.ranking_result = ComparisonReportRankingResult::Improved;
                    }
                    ComparisonResult::Regressed => {
                        mean_diff = red(mean_diff).to_string();
                        benchmark_new_mean_str = red(benchmark_new_mean_str).to_string();
                        benchmark_old_mean_str = green(bold(benchmark_old_mean_str)).to_string();
                        function_id_new_color_str = red(function_id_new_color_str).to_string();
                        function_id_old_color_str =
                            green(bold(function_id_old_color_str)).to_string();
                        explanation_str = format!(
                            "Performance has {}",
                            red(bold(format!("regressed {mean_diff_pct_str}")))
                        );
                        // comparison_report_results.push(ComparisonReportRanking {
                        //     function_id_new: function_id_new_str,
                        //     function_id_old: function_id_old_str,
                        //     result: ComparisonReportRankingResult::Regressed,
                        // });
                        comparison.ranking_result = ComparisonReportRankingResult::Regressed;
                    }
                    ComparisonResult::NonSignificant => {
                        mean_diff = faint(bold(mean_diff)).to_string();
                        if mean_diff_point_estimate < 0.0 {
                            benchmark_new_mean_str =
                                faint(bold(benchmark_new_mean_str)).to_string();
                            function_id_new_color_str =
                                faint(bold(function_id_new_color_str)).to_string();
                            explanation_str = format!(
                                "Improved {} within noise threshold of ±{:.2}%",
                                faint(bold(mean_diff_pct_str)),
                                noise_threshold * 1e2
                            );
                            // comparison_report_results.push(ComparisonReportRanking {
                            //     function_id_new: function_id_new_str,
                            //     function_id_old: function_id_old_str,
                            //     result: ComparisonReportRankingResult::NonSignificantImproved,
                            // });
                            comparison.ranking_result =
                                ComparisonReportRankingResult::NonSignificantImproved;
                        } else {
                            benchmark_old_mean_str =
                                faint(bold(benchmark_old_mean_str)).to_string();
                            function_id_old_color_str =
                                faint(bold(function_id_old_color_str)).to_string();
                            explanation_str = format!(
                                "Regressed {} within noise threshold of ±{:.2}%",
                                faint(bold(mean_diff_pct_str)),
                                noise_threshold * 1e2
                            );
                            // comparison_report_results.push(ComparisonReportRanking {
                            //     function_id_new: function_id_new_str,
                            //     function_id_old: function_id_old_str,
                            //     result: ComparisonReportRankingResult::NonSignificantRegressed,
                            // });
                            comparison.ranking_result =
                                ComparisonReportRankingResult::NonSignificantImproved;
                        }
                    }
                }
            } else {
                explanation_str = "No change in performance detected".to_owned();
                // comparison_report_results.push(ComparisonReportRanking {
                //     function_id_new: function_id_new_str,
                //     function_id_old: function_id_old_str,
                //     result: ComparisonReportRankingResult::NoChange,
                // });
            }

            changes_table_rows.push(ChangesTable {
                function_id_vs: format!(
                    "{} vs {}",
                    &function_id_old_color_str, &function_id_new_color_str
                ),
                latency_mean: format!("{} vs {}", &benchmark_old_mean_str, &benchmark_new_mean_str),
                latency_mean_change: format!(
                    "{} [{:+.2},{:+.2}] {}% CI (p = {} {} {})",
                    &mean_diff,
                    mean_diff_ci_lower_bound,
                    mean_diff_ci_upper_bound,
                    // (mean_diff_ci.confidence_level * 1000.0) / 10.0,
                    (mean_diff_ci.confidence_level * 100.0),
                    p_value_formatter.fmt(comp.p_value),
                    if is_mean_different { "<" } else { ">" },
                    &significance_threshold
                ),
                result: explanation_str,
            });
        }

        let ranking: RankingResult = rank_fastest_with_scores(my_comparisons_report);
        // eprintln!("rank_fastest_with_scores: {ranking:?}");
        let mut ranking_table_rows: Vec<RankingTable> = Vec::with_capacity(12);
        for (idx, functions) in ranking.ranks.iter().enumerate() {
            struct RankTempData {
                function_id: String,
                latency_mean_str: String,
                latency_mean: f64,
                latency_mean_ci: ConfidenceInterval,
            }
            let mut rank_temp: Vec<RankTempData> = Vec::with_capacity(12);
            for function in functions {
                if let Some(data) = functions_comparison_report_data.get(function) {
                    rank_temp.push(RankTempData {
                        function_id: function.clone(),
                        latency_mean_str: data.latency_mean_str.clone(),
                        latency_mean: data.latency_mean,
                        latency_mean_ci: data.latency_mean_ci.clone(),
                    });
                }
            }

            rank_temp.sort_by(|a, b| a.latency_mean.partial_cmp(&b.latency_mean).unwrap());
            // let min_latency_mean = rank_temp.first().unwrap().latency_mean;
            let mut min_latency_mean: f64 = 1.0;
            // if idx == 0 {
            //     min_latency_mean = r.latency_mean;
            // }
            for r in &rank_temp {
                // for (i, r) in rank_temp.iter().enumerate() {
                if idx == 0 {
                    ranking_table_rows.push(RankingTable {
                        ranking: idx + 1,
                        function_id: r.function_id.clone(),
                        latency_mean: format!(
                            "{} [{:.2},{:.2}] {}% CI",
                            r.latency_mean_str,
                            r.latency_mean_ci.lower_bound,
                            r.latency_mean_ci.upper_bound,
                            // (r.latency_mean_ci.confidence_level * 1000.0) / 10.0,
                            (r.latency_mean_ci.confidence_level * 100.0),
                        ),
                        // relative_performance: "1x".to_string(),
                        relative_performance: String::new(),
                    });
                } else {
                    let ratio_to_baseline: f64 = r.latency_mean / min_latency_mean;
                    ranking_table_rows.push(RankingTable {
                        ranking: idx + 1,
                        function_id: r.function_id.clone(),
                        latency_mean: format!(
                            "{} [{:.2},{:.2}] {}% CI",
                            r.latency_mean_str,
                            r.latency_mean_ci.lower_bound,
                            r.latency_mean_ci.upper_bound,
                            // (r.latency_mean_ci.confidence_level * 1000.0) / 10.0,
                            (r.latency_mean_ci.confidence_level * 100.0),
                        ),
                        relative_performance: format!(
                            "{:.2}x increase in execution time ({:.2}%)",
                            ratio_to_baseline,
                            (ratio_to_baseline - 1.0) * 100.0
                        ),
                    });
                }
            }
        }

        GroupComparisonTables {
            changes_table_rows,
            ranking_table_rows,
        }
    }

    pub fn print_tables(&self) {
        eprintln!("{}", self.comparison_tables);
    }
}

#[derive(Tabled)]
pub struct ChangesTable {
    pub function_id_vs: String,
    #[tabled(rename = "Latency (mean)")]
    pub latency_mean: String,
    #[tabled(rename = "Latency Change (mean)")]
    pub latency_mean_change: String,
    #[tabled(rename = "Result")]
    pub result: String,
}

#[derive(Tabled, Debug)]
pub struct RankingTable {
    #[tabled(rename = "Ranking")]
    pub ranking: usize,
    #[tabled(rename = "Function")]
    pub function_id: String,
    #[tabled(rename = "Latency (mean)")]
    pub latency_mean: String,
    #[tabled(rename = "Relative Performance")]
    pub relative_performance: String,
}

#[derive(Debug)]
pub struct MergeDuplicatesVerticalFirst;

impl<R, D> TableOption<R, ColoredConfig, D> for MergeDuplicatesVerticalFirst
where
    R: Records + PeekableRecords + ExactRecords,
{
    #[allow(clippy::assigning_clones)]
    fn change(self, records: &mut R, cfg: &mut ColoredConfig, _: &mut D) {
        let count_rows = records.count_rows();
        let count_cols = records.count_columns();

        if count_rows == 0 || count_cols == 0 {
            return;
        }

        // for column in 0..count_cols {
        let mut repeat_length = 0;
        let mut repeat_value = String::with_capacity(8);
        let mut repeat_is_set = false;
        let mut last_is_row_span = false;
        for row in (0..count_rows).rev() {
            if last_is_row_span {
                last_is_row_span = false;
                continue;
            }

            let is_cell_visible = cfg.is_cell_visible((row, 0).into());
            let is_row_span_cell = cfg.get_column_span((row, 0).into()).is_some();

            if !repeat_is_set {
                if !is_cell_visible {
                    continue;
                }

                if is_row_span_cell {
                    continue;
                }

                repeat_length = 1;
                repeat_value = records.get_text((row, 0).into()).to_owned();
                repeat_is_set = true;
                continue;
            }

            if is_row_span_cell {
                repeat_is_set = false;
                last_is_row_span = true;
                continue;
            }

            if !is_cell_visible {
                repeat_is_set = false;
                continue;
            }

            let text = records.get_text((row, 0).into());
            let is_duplicate = text == repeat_value;

            if is_duplicate {
                repeat_length += 1;
                continue;
            }

            if repeat_length > 1 {
                cfg.set_row_span((row + 1, 0).into(), repeat_length);
            }

            repeat_length = 1;
            repeat_value = records.get_text((row, 0).into()).to_owned();
        }

        if repeat_length > 1 {
            cfg.set_row_span((0, 0).into(), repeat_length);
        }
        // }
    }
}
