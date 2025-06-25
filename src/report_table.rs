use crate::estimate::{ConfidenceInterval, Estimates};
use crate::format;
use crate::model::{Benchmark, Model};
use crate::report::{
    compare_to_threshold, rank_fastest_with_scores, BenchmarkId, ComparisonReport,
    ComparisonReportRanking, ComparisonReportRankingData, ComparisonReportRankingResult,
    ComparisonResult, OwnedMeasurementData,
};
use crate::value_formatter::ValueFormatter;
use itertools::Itertools;
use tabled::{
    grid::config::ColoredConfig,
    grid::records::{ExactRecords, PeekableRecords, Records},
    settings::{style::Style, themes::BorderCorrection, Alignment, Format, TableOption},
    Table, Tabled,
};

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut};

pub struct GroupsComparisons(HashMap<String, GroupComparisonTables>);

impl GroupsComparisons {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

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
        for (group, comparison_tables) in &**self {
            let mut changes_table = Table::new(&comparison_tables.changes_table_rows);
            changes_table.modify((0, 0), Format::content(|_| group.to_string()));
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
    group_id: String,
    changes_table_rows: Vec<ChangesTable>,
    ranking_table_rows: Vec<RankingTable>,
}

pub struct IntraGroupComparison {
    comparison_tables: GroupsComparisons,
}

impl IntraGroupComparison {
    pub fn new() -> Self {
        Self {
            comparison_tables: GroupsComparisons::with_capacity(12),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    fn green(&self, s: String) -> String {
        format!("\x1B[32m{s}\x1B[39m")
    }

    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    fn yellow(&self, s: String) -> String {
        format!("\x1B[33m{s}\x1B[39m")
    }

    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    fn red(&self, s: String) -> String {
        format!("\x1B[31m{s}\x1B[39m")
    }

    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    fn bold(&self, s: String) -> String {
        format!("\x1B[1m{s}\x1B[22m")
    }

    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    fn faint(&self, s: String) -> String {
        format!("\x1B[2m{s}\x1B[22m")
    }

    pub fn get_intra_group_comparison_data(&mut self, model: &Model, formatter: &ValueFormatter) {
        for (group_id, benchmark_group) in &model.groups {
            let e = self.comparison_tables.entry(group_id.to_string());
            if let Entry::Occupied(mut entry) = e {
                eprintln!("\nget_intra_group_comparison_data - group_id: {group_id} ");
                let mut comparisons: Vec<ComparisonReport> = Vec::with_capacity(12);
                for combinations in benchmark_group
                    .benchmarks
                    .iter()
                    .tuple_combinations::<((&BenchmarkId, &Benchmark), (&BenchmarkId, &Benchmark))>(
                    )
                {
                    let ((id_new, benchmark_new), (id_old, benchmark_old)): (
                        (&BenchmarkId, &Benchmark),
                        (&BenchmarkId, &Benchmark),
                    ) = combinations;
                    let comp = crate::analysis::analysis_comparison(
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

                    comparisons.push(ComparisonReport {
                        id_new,
                        id_old,
                        benchmark_new,
                        benchmark_old,
                        comp,
                    });
                }

                if !comparisons.is_empty() {
                    eprintln!("\nget_intra_group_comparison_data - group_id: {group_id} ");
                    let k = self.parse_comparisons(group_id, &comparisons, formatter);
                    entry.insert(k);
                }
                drop(comparisons);
            }
        }
    }

    fn parse_comparisons(
        &self,
        group_id: &str,
        comparisons: &Vec<ComparisonReport>,
        formatter: &ValueFormatter,
    ) -> GroupComparisonTables {
        // self.text_overwrite();

        let mut comparison_report_results: Vec<ComparisonReportRanking> = Vec::with_capacity(12);
        let mut p_value_formatters: HashMap<format::FloatKey, format::PValueFormatter> =
            HashMap::with_capacity(12);
        let mut changes_table_rows: Vec<ChangesTable> = Vec::with_capacity(12);

        let mut functions_comparison_report_data: HashMap<String, ComparisonReportRankingData> =
            HashMap::with_capacity(12);

        for comparison in comparisons {
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
                        mean_diff = self.green(self.bold(mean_diff));
                        benchmark_new_mean_str = self.green(self.bold(benchmark_new_mean_str));
                        benchmark_old_mean_str = self.red(benchmark_old_mean_str);
                        function_id_new_color_str =
                            self.green(self.bold(function_id_new_color_str));
                        function_id_old_color_str = self.red(function_id_old_color_str);
                        explanation_str = format!(
                            "Performance has {}",
                            self.green(self.bold(format!("improved {mean_diff_pct_str}")))
                        );
                        comparison_report_results.push(ComparisonReportRanking {
                            function_id_new: function_id_new_str,
                            function_id_old: function_id_old_str,
                            result: ComparisonReportRankingResult::Improved,
                        });
                    }
                    ComparisonResult::Regressed => {
                        mean_diff = self.red(mean_diff);
                        benchmark_new_mean_str = self.red(benchmark_new_mean_str);
                        benchmark_old_mean_str = self.green(self.bold(benchmark_old_mean_str));
                        function_id_new_color_str = self.red(function_id_new_color_str);
                        function_id_old_color_str =
                            self.green(self.bold(function_id_old_color_str));
                        explanation_str = format!(
                            "Performance has {}",
                            self.red(self.bold(format!("regressed {mean_diff_pct_str}")))
                        );
                        comparison_report_results.push(ComparisonReportRanking {
                            function_id_new: function_id_new_str,
                            function_id_old: function_id_old_str,
                            result: ComparisonReportRankingResult::Regressed,
                        });
                    }
                    ComparisonResult::NonSignificant => {
                        mean_diff = self.faint(self.bold(mean_diff));
                        if mean_diff_point_estimate < 0.0 {
                            benchmark_new_mean_str = self.faint(self.bold(benchmark_new_mean_str));
                            function_id_new_color_str =
                                self.faint(self.bold(function_id_new_color_str));
                            explanation_str = format!(
                                "Improved {} within noise threshold of ±{:.2}%",
                                self.faint(self.bold(mean_diff_pct_str)),
                                noise_threshold * 1e2
                            );
                            comparison_report_results.push(ComparisonReportRanking {
                                function_id_new: function_id_new_str,
                                function_id_old: function_id_old_str,
                                result: ComparisonReportRankingResult::NonSignificantImproved,
                            });
                        } else {
                            benchmark_old_mean_str = self.faint(self.bold(benchmark_old_mean_str));
                            function_id_old_color_str =
                                self.faint(self.bold(function_id_old_color_str));
                            explanation_str = format!(
                                "Regressed {} within noise threshold of ±{:.2}%",
                                self.faint(self.bold(mean_diff_pct_str)),
                                noise_threshold * 1e2
                            );
                            comparison_report_results.push(ComparisonReportRanking {
                                function_id_new: function_id_new_str,
                                function_id_old: function_id_old_str,
                                result: ComparisonReportRankingResult::NonSignificantRegressed,
                            });
                        }
                    }
                }
            } else {
                explanation_str = "No change in performance detected".to_owned();
                comparison_report_results.push(ComparisonReportRanking {
                    function_id_new: function_id_new_str,
                    function_id_old: function_id_old_str,
                    result: ComparisonReportRankingResult::NoChange,
                });
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
                    (mean_diff_ci.confidence_level * 1000.0) / 10.0,
                    p_value_formatter.fmt(comp.p_value),
                    if is_mean_different { "<" } else { ">" },
                    &significance_threshold
                ),
                result: explanation_str,
            });
        }

        // print_changes_table(group_id, &changes_table_rows);

        let ranking = rank_fastest_with_scores(&comparison_report_results);
        // eprintln!("1 ranking: {ranking:?}");
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
            for r in &rank_temp {
                ranking_table_rows.push(RankingTable {
                    ranking: idx + 1,
                    function_id: r.function_id.clone(),
                    latency_mean: format!(
                        "{} [{:.2},{:.2}] {}% CI",
                        r.latency_mean_str,
                        r.latency_mean_ci.lower_bound,
                        r.latency_mean_ci.upper_bound,
                        (r.latency_mean_ci.confidence_level * 1000.0) / 10.0,
                    ),
                });
            }
        }
        // eprintln!("2 ranking_table_rows: {ranking_table_rows:?}");
        // print_ranking_table(group_id, &ranking_table_rows);

        GroupComparisonTables {
            group_id: group_id.to_owned(),
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

// pub fn print_changes_table(group_id: &str, rows: &[ChangesTable]) {
//     let mut table = Table::new(rows);

//     table.modify((0, 0), Format::content(|_| group_id.to_string()));

//     table
//         // .with(Style::modern_rounded())
//         .with(Style::modern())
//         // .with(MergeDuplicatesVerticalFirst)
//         // .with(BorderCorrection::span())
//         .with(Alignment::center())
//         .with(Alignment::center_vertical());

//     eprintln!("{table}");
// }

// pub fn print_ranking_table(group_id: &str, rows: &[RankingTable]) {
//     let mut table = Table::new(rows);

//     table
//         // .with(Style::modern_rounded())
//         .with(Style::modern())
//         .with(MergeDuplicatesVerticalFirst)
//         .with(BorderCorrection::span())
//         .with(Alignment::center())
//         .with(Alignment::center_vertical());

//     eprintln!("{table}");
// }
