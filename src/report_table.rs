use crate::estimate::{ConfidenceInterval, Estimate, Estimates};
use crate::format;
use crate::model::{Benchmark, BenchmarkGroup};
use crate::report::{
    compare_to_threshold, BenchmarkId, ComparisonData, ComparisonResult, OwnedMeasurementData,
};
use crate::value_formatter::ValueFormatter;
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

pub struct ComparisonReport<'benchmark_group> {
    pub id_new: &'benchmark_group BenchmarkId,
    pub id_old: &'benchmark_group BenchmarkId,
    pub benchmark_new: &'benchmark_group Benchmark,
    pub benchmark_old: &'benchmark_group Benchmark,
    pub comp: ComparisonData,
    pub comp_result: ComparisonReportResult,
}

impl<'benchmark_group> ComparisonReport<'benchmark_group> {
    const fn new(
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
            comp_result: ComparisonReportResult::NoChange,
        }
    }
}

#[derive(Debug)]
pub enum ComparisonReportResult {
    Improved,                // +2 score for new, -2 score for old
    Regressed,               // +2 score for old, -2 score for new
    NonSignificantImproved,  // +1 score for new, -1 score for old
    NonSignificantRegressed, // +1 score for old, -1 score for new
    NoChange,                //  0 score
}

#[derive(Debug, Clone)]
struct BenchmarkExtended<'benchmark_group> {
    pub id: &'benchmark_group BenchmarkId,
    pub benchmark: &'benchmark_group Benchmark,
    pub latency_mean_str: String,
    pub latency_mean: f64,
    pub latency_mean_ci: ConfidenceInterval,
    pub score: i32,
}

impl<'benchmark_group> BenchmarkExtended<'benchmark_group> {
    const fn new(
        id: &'benchmark_group BenchmarkId,
        benchmark: &'benchmark_group Benchmark,
        latency_mean_str: String,
        latency_mean: f64,
        latency_mean_ci: ConfidenceInterval,
    ) -> Self {
        Self {
            id,
            benchmark,
            latency_mean_str,
            latency_mean,
            latency_mean_ci,
            score: 0,
        }
    }
}

/// Fast-to-slow ranking plus scores.
#[derive(Debug)]
pub struct RankingResultExtended<'benchmark_group> {
    ranks: Vec<Vec<BenchmarkExtended<'benchmark_group>>>,
}

fn rank_fastest_with_scores<'my_comparisons_report, 'benchmark_group>(
    comparisons_report: &'my_comparisons_report [ComparisonReport<'benchmark_group>],
    mut benchmark_extended_data: HashMap<String, BenchmarkExtended<'benchmark_group>>,
) -> RankingResultExtended<'benchmark_group> {
    use std::cmp::Ordering;

    // ------------------------------------------------------
    // 1. Accumulate score deltas WITHOUT touching the heavy
    // `benchmark_extended_data` map on every comparison.
    // ------------------------------------------------------
    let mut score_deltas: HashMap<&str, i32> =
        HashMap::with_capacity(benchmark_extended_data.len() * 2);

    for report in comparisons_report {
        let id_new: &String = report.id_new.function_id.as_ref().unwrap();
        let id_old: &String = report.id_old.function_id.as_ref().unwrap();

        let (delta_new, delta_old) = match report.comp_result {
            ComparisonReportResult::Improved => (2, -2),
            ComparisonReportResult::Regressed => (-2, 2),
            ComparisonReportResult::NonSignificantImproved => (1, -1),
            ComparisonReportResult::NonSignificantRegressed => (-1, 1),
            ComparisonReportResult::NoChange => (0, 0),
        };

        if delta_new != 0 {
            *score_deltas.entry(id_new).or_default() += delta_new;
        }
        if delta_old != 0 {
            *score_deltas.entry(id_old).or_default() += delta_old;
        }
    }

    // ------------------------------------------------------------
    // 2. Apply all deltas in one sequential pass over the big map.
    // ------------------------------------------------------------
    for (id, delta) in score_deltas {
        if let Some(be) = benchmark_extended_data.get_mut(id) {
            be.score += delta;
        }
    }

    // -------------------------------------------------------
    // 3. Collect, sort and group the benchmarks.
    // - primary key : descending score (higher == faster)
    // - secondary : ascending latency_mean (lower == faster)
    // - tertiary : alphabetical function_id for stability
    // -------------------------------------------------------
    let mut all: Vec<BenchmarkExtended<'benchmark_group>> =
        Vec::with_capacity(benchmark_extended_data.len());
    all.extend(benchmark_extended_data.into_values());

    all.sort_unstable_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| {
                a.latency_mean
                    .partial_cmp(&b.latency_mean)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| a.id.function_id.cmp(&b.id.function_id))
    });

    // ----------------------------------------------------------------
    // 4. Convert the flat, ordered list into rank buckets where every
    // bucket corresponds to a distinct score.
    // ----------------------------------------------------------------
    let mut ranks: Vec<Vec<BenchmarkExtended<'benchmark_group>>> = Vec::with_capacity(all.len());

    // This manual grouping over an iterator is efficient and clear.
    // It processes the sorted values in a single pass.
    let mut group_iter: std::vec::IntoIter<BenchmarkExtended<'benchmark_group>> = all.into_iter();
    // The first element always starts a new group.
    let first_be: BenchmarkExtended<'benchmark_group> = group_iter.next().unwrap();
    let mut current_score: i32 = first_be.score;
    let mut current_group: Vec<BenchmarkExtended<'benchmark_group>> = vec![first_be];

    for be in group_iter {
        if be.score == current_score {
            current_group.push(be);
        } else {
            ranks.push(current_group);
            current_score = be.score;
            current_group = vec![be];
        }
    }
    // Add the last group to the ranks.
    ranks.push(current_group);

    RankingResultExtended::<'benchmark_group> { ranks }
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
const fn bold<T: fmt::Display>(s: T) -> Bold<T> {
    Bold(s)
}

struct Green<T>(T);

impl<T: fmt::Display> fmt::Display for Green<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[32m{}\x1B[39m", self.0)
    }
}

#[inline]
const fn green<T: fmt::Display>(s: T) -> Green<T> {
    Green(s)
}

struct Red<T>(T);

impl<T: fmt::Display> fmt::Display for Red<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[31m{}\x1B[39m", self.0)
    }
}

#[inline]
const fn red<T: fmt::Display>(s: T) -> Red<T> {
    Red(s)
}

struct Faint<T>(T);

impl<T: fmt::Display> fmt::Display for Faint<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1B[2m{}\x1B[22m", self.0)
    }
}

#[inline]
const fn faint<T: fmt::Display>(s: T) -> Faint<T> {
    Faint(s)
}

impl IntraGroupComparison {
    pub fn new() -> Self {
        Self {
            comparison_tables: GroupsComparisons::with_capacity(12),
        }
    }

    pub fn get_intra_group_comparison_data<'group_id, 'benchmark_group, 'formatter>(
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
                                                crate::analysis::MeasuredValues::<'_> {
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
                                                        crate::analysis::MeasuredValues::<'_> {
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
        }

        if !comparisons_report.is_empty() {
            self.parse_comparisons(group_id, &mut comparisons_report, formatter);
        }
    }

    fn parse_comparisons<'group_id, 'my_comparisons_report, 'benchmark_group, 'formatter>(
        &mut self,
        group_id: &'group_id str,
        my_comparisons_report: &'my_comparisons_report mut [ComparisonReport<'benchmark_group>],
        formatter: &'formatter ValueFormatter,
    ) {
        let mut p_value_formatters: HashMap<format::FloatKey, format::PValueFormatter> =
            HashMap::with_capacity(12);
        let mut changes_table_rows: Vec<ChangesTable> = Vec::with_capacity(12);

        let mut benchmark_extended_data: HashMap<String, BenchmarkExtended<'benchmark_group>> =
            HashMap::with_capacity(12);

        for comparison in &mut *my_comparisons_report {
            let comp: &ComparisonData = &comparison.comp;
            let significance_threshold: f64 = comp.significance_threshold;
            let is_mean_different: bool = comp.p_value < significance_threshold;
            let mean_diff_est: &Estimate = &comp.relative_estimates.mean;
            let mean_diff_point_estimate: f64 = mean_diff_est.point_estimate;

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

            let mean_diff_ci: &ConfidenceInterval = &mean_diff_est.confidence_interval;
            let mean_diff_ci_lower_bound: f64 = mean_diff_ci.lower_bound * benchmark_old_mean;
            let mean_diff_ci_upper_bound: f64 = mean_diff_ci.upper_bound * benchmark_old_mean;
            let mean_diff_pct_str: String = format!("{:.2}%", mean_diff_point_estimate.abs() * 1e2);
            let noise_threshold: f64 = comp.noise_threshold;
            let function_id_old_str: String =
                comparison.id_old.function_id.as_ref().unwrap().to_owned();
            let function_id_new_str: String =
                comparison.id_new.function_id.as_ref().unwrap().to_owned();
            let explanation_str: String;

            let p_value_formatter: &mut format::PValueFormatter = p_value_formatters
                .entry(format::FloatKey(comp.p_value))
                .or_insert_with(|| format::PValueFormatter::new(significance_threshold));
            let mut mean_diff: String =
                format!("{:+.2} ns", mean_diff_point_estimate * benchmark_old_mean);
            let mut function_id_old_color_str = function_id_old_str.clone();
            let mut function_id_new_color_str = function_id_new_str.clone();
            let mut benchmark_old_mean_str = formatter.format_value(benchmark_old_mean);
            let mut benchmark_new_mean_str = formatter.format_value(benchmark_new_mean);
            benchmark_extended_data.insert(
                function_id_new_str.clone(),
                BenchmarkExtended::<'benchmark_group>::new(
                    comparison.id_new,
                    comparison.benchmark_new,
                    benchmark_new_mean_str.clone(),
                    benchmark_new_mean,
                    benchmark_new_mean_ci.clone(),
                ),
            );
            benchmark_extended_data.insert(
                function_id_old_str.clone(),
                BenchmarkExtended::<'benchmark_group>::new(
                    comparison.id_old,
                    comparison.benchmark_old,
                    benchmark_old_mean_str.clone(),
                    benchmark_old_mean,
                    benchmark_old_mean_ci.clone(),
                ),
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
                        comparison.comp_result = ComparisonReportResult::Improved;
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
                        comparison.comp_result = ComparisonReportResult::Regressed;
                    }
                    ComparisonResult::NonSignificant => {
                        mean_diff = faint(bold(mean_diff)).to_string();
                        if mean_diff_point_estimate.is_sign_negative() {
                            benchmark_new_mean_str =
                                faint(bold(benchmark_new_mean_str)).to_string();
                            function_id_new_color_str =
                                faint(bold(function_id_new_color_str)).to_string();
                            explanation_str = format!(
                                "Improved {} within noise threshold of ±{:.2}%",
                                faint(bold(mean_diff_pct_str)),
                                noise_threshold * 1e2
                            );
                            comparison.comp_result = ComparisonReportResult::NonSignificantImproved;
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
                            comparison.comp_result =
                                ComparisonReportResult::NonSignificantRegressed;
                        }
                    }
                }
            } else {
                explanation_str = "No change in performance detected".to_owned();
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

        let ranking: RankingResultExtended<'benchmark_group> =
            rank_fastest_with_scores(my_comparisons_report, benchmark_extended_data);

        for r in &ranking.ranks {
            eprintln!(
                "Rank: {} - Functions: {}",
                r.len(),
                r.iter()
                    .map(|b| b.id.function_id.as_ref().unwrap())
                    .join(", ")
            );
        }

        let mut ranking_table_rows: Vec<RankingTable> = Vec::with_capacity(12);

        // for rank in &ranking.ranks {}
        for (i, rank) in ranking.ranks.iter().enumerate() {
            for benchmark_extented in rank {
                ranking_table_rows.push(RankingTable {
                    ranking: i + 1,
                    function_id: benchmark_extented.id.function_id.as_ref().unwrap().clone(),
                    // function_id: benchmark_extented.id.function_id.unwrap().clone(),
                    latency_mean: format!(
                        "{} [{:.2},{:.2}] {}% CI",
                        benchmark_extented.latency_mean_str,
                        benchmark_extented.latency_mean_ci.lower_bound,
                        benchmark_extented.latency_mean_ci.upper_bound,
                        // (r.latency_mean_ci.confidence_level * 1000.0) / 10.0,
                        (benchmark_extented.latency_mean_ci.confidence_level * 100.0),
                    ),
                    relative_performance: format!("test"),
                    // relative_performance: format!(
                    //     "{:.2}x increase in execution time ({:.2}%)",
                    //     ratio_to_baseline,
                    //     (ratio_to_baseline - 1.0) * 100.0
                    // ),
                });
            }
        }

        // for (idx, functions) in ranking.ranks.iter().enumerate() {
        //     struct RankTempData {
        //         function_id: String,
        //         latency_mean_str: String,
        //         latency_mean: f64,
        //         latency_mean_ci: ConfidenceInterval,
        //     }
        //     let mut rank_temp: Vec<RankTempData> = Vec::with_capacity(12);
        //     for function in functions {
        //         if let Some(data) = functions_comparison_report_data.get(function) {
        //             rank_temp.push(RankTempData {
        //                 function_id: function.clone(),
        //                 latency_mean_str: data.latency_mean_str.clone(),
        //                 latency_mean: data.latency_mean,
        //                 latency_mean_ci: data.latency_mean_ci.clone(),
        //             });
        //         }
        //     }

        //     rank_temp.sort_by(|a, b| a.latency_mean.partial_cmp(&b.latency_mean).unwrap());
        //     // let min_latency_mean = rank_temp.first().unwrap().latency_mean;
        //     let mut min_latency_mean: f64 = 1.0;
        //     for r in &rank_temp {
        //         if idx == 0 {
        //             min_latency_mean = r.latency_mean;
        //             ranking_table_rows.push(RankingTable {
        //                 ranking: idx + 1,
        //                 function_id: r.function_id.clone(),
        //                 latency_mean: format!(
        //                     "{} [{:.2},{:.2}] {}% CI",
        //                     r.latency_mean_str,
        //                     r.latency_mean_ci.lower_bound,
        //                     r.latency_mean_ci.upper_bound,
        //                     // (r.latency_mean_ci.confidence_level * 1000.0) / 10.0,
        //                     (r.latency_mean_ci.confidence_level * 100.0),
        //                 ),
        //                 // relative_performance: "1x".to_string(),
        //                 relative_performance: String::new(),
        //             });
        //         } else {
        //             let ratio_to_baseline: f64 = r.latency_mean / min_latency_mean;
        // ranking_table_rows.push(RankingTable {
        //     ranking: idx + 1,
        //     function_id: r.function_id.clone(),
        //     latency_mean: format!(
        //         "{} [{:.2},{:.2}] {}% CI",
        //         r.latency_mean_str,
        //         r.latency_mean_ci.lower_bound,
        //         r.latency_mean_ci.upper_bound,
        //         // (r.latency_mean_ci.confidence_level * 1000.0) / 10.0,
        //         (r.latency_mean_ci.confidence_level * 100.0),
        //     ),
        //     relative_performance: format!(
        //         "{:.2}x increase in execution time ({:.2}%)",
        //         ratio_to_baseline,
        //         (ratio_to_baseline - 1.0) * 100.0
        //     ),
        // });
        //         }
        //     }
        // }

        if let Some(_) = self.comparison_tables.insert(
            group_id.to_owned(),
            GroupComparisonTables {
                changes_table_rows,
                ranking_table_rows,
            },
        ) {
            eprintln!("ALREADY INSERTED: {group_id}");
        } else {
            eprintln!("NOT INSERTED: {group_id}");
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
