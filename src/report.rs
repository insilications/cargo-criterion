use crate::connection::{PlotConfiguration, Throughput};
use crate::estimate::{ChangeDistributions, ChangeEstimates, Distributions, Estimate, Estimates};
use crate::format;
use crate::model::{Benchmark, BenchmarkGroup, Model, SavedStatistics};
use crate::stats::bivariate::regression::Slope;
use crate::stats::bivariate::Data;
use crate::stats::univariate::outliers::tukey::LabeledSample;
use crate::stats::univariate::Sample;
use crate::stats::Distribution;
use crate::value_formatter::ValueFormatter;
use cli_table::{
    format::{Align, Border, HorizontalLine, Justify, Separator, VerticalLine},
    print_stderr, Cell as TableCell, CellStruct, Color, Style, Table,
};
use std::cell::Cell;
use std::cmp;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::io::stderr;
use std::io::Write;
use std::path::{Path, PathBuf};
use union_find::{QuickUnionUf, UnionByRank, UnionFind};

const MAX_DIRECTORY_NAME_LEN: usize = 64;
const MAX_TITLE_LEN: usize = 100;

pub struct ComparisonReport<'a> {
    pub id_new: &'a BenchmarkId,
    pub id_old: &'a BenchmarkId,
    pub benchmark_new: &'a Benchmark,
    pub benchmark_old: &'a Benchmark,
    pub comp: ComparisonData,
}

#[derive(Debug)]
pub enum ComparisonReportRankingResult {
    Improved,
    Regressed,
    NonSignificant,
    // NoChange,
}

#[derive(Debug)]
pub struct ComparisonReportRanking {
    pub function_id_new: String,
    pub function_id_old: String,
    pub result: ComparisonReportRankingResult,
}

/// A simple Disjoint Set Union (DSU) data structure.
/// It uses path compression and union by rank (implicitly by map structure) for efficiency.
struct DisjointSet<'a> {
    parent: HashMap<&'a str, &'a str>,
}

impl<'a> DisjointSet<'a> {
    /// Creates a new DSU where each item is its own parent initially.
    fn new(items: impl IntoIterator<Item = &'a str>) -> Self {
        let parent = items.into_iter().map(|item| (item, item)).collect();
        DisjointSet { parent }
    }

    /// Finds the representative of the set containing `item`.
    /// Implements path compression for optimization.
    fn find(&mut self, item: &'a str) -> &'a str {
        let parent = self.parent.get(item).unwrap();
        if parent == &item {
            return item;
        }
        let representative = self.find(parent);
        self.parent.insert(item, representative);
        representative
    }

    /// Merges the sets containing `item1` and `item2`.
    fn union(&mut self, item1: &'a str, item2: &'a str) {
        let root1 = self.find(item1);
        let root2 = self.find(item2);
        if root1 != root2 {
            self.parent.insert(root1, root2);
        }
    }
}

/// Ranks function IDs based on a series of comparison reports.
///
/// The ranking is determined by performing a topological sort on a graph of the functions.
/// Tied functions (`NonSignificant``) are grouped together at the same rank.
///
/// # Returns
/// A `Result` containing either:
/// - `Ok(Vec<Vec<String>>)`: A list of rank groups, ordered from fastest to slowest.
///   Each inner vector contains function IDs that are tied in performance.
/// - `Err(String)`: An error message if the reports are contradictory (contain a cycle).
pub fn rank_functions(reports: &[ComparisonReportRanking]) -> Result<Vec<Vec<String>>, String> {
    // --- 1. Collect all unique function IDs ---
    // This HashSet owns all the strings, allowing us to use &str slices
    // in subsequent data structures to avoid cloning strings repeatedly.
    let mut unique_ids = HashSet::new();
    for report in reports {
        unique_ids.insert(report.function_id_new.clone());
        unique_ids.insert(report.function_id_old.clone());
    }
    let unique_ids_refs: HashSet<&str> = unique_ids.iter().map(AsRef::as_ref).collect();

    // --- 2. Group tied functions using DSU ---
    let mut dsu = DisjointSet::new(unique_ids_refs.iter().copied());
    for report in reports {
        if let ComparisonReportRankingResult::NonSignificant = report.result {
            dsu.union(&report.function_id_new, &report.function_id_old);
        }
    }

    // --- 3. Build the graph of equivalence classes (rank groups) ---
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut in_degree: HashMap<&str, usize> = HashMap::new();

    // Initialize graph nodes (representatives of each set)
    for &id in &unique_ids_refs {
        let root = dsu.find(id);
        adj.entry(root).or_default();
        in_degree.entry(root).or_insert(0);
    }

    for report in reports {
        let rep_new = dsu.find(&report.function_id_new);
        let rep_old = dsu.find(&report.function_id_old);

        // Only add edges between different sets
        if rep_new == rep_old {
            continue;
        }

        // An edge `u -> v` means `u` is faster than `v`.
        let (faster, slower) = match report.result {
            ComparisonReportRankingResult::Improved => (rep_new, rep_old),
            ComparisonReportRankingResult::Regressed => (rep_old, rep_new),
            ComparisonReportRankingResult::NonSignificant => continue,
        };

        if adj.entry(faster).or_default().insert(slower) {
            *in_degree.entry(slower).or_insert(0) += 1;
        }
    }

    // --- 4. Perform Topological Sort (Kahn's Algorithm) ---
    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
        .collect();

    let mut sorted_reps = Vec::new();
    while let Some(u) = queue.pop_front() {
        sorted_reps.push(u);
        // We must clone the neighbors to avoid borrowing issues with `in_degree`.
        let neighbors = adj.get(u).cloned().unwrap_or_default();
        for v in neighbors {
            if let Some(degree) = in_degree.get_mut(v) {
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    // --- 5. Check for cycles and format the output ---
    if sorted_reps.len() != adj.len() {
        return Err(
            "Contradictory reports found (cycle detected in performance graph)".to_string(),
        );
    }

    // Group all functions by their representative to form the final ranked list.
    let mut groups: HashMap<&str, Vec<String>> = HashMap::new();
    for id in &unique_ids {
        let root = dsu.find(id);
        groups.entry(root).or_default().push(id.clone());
    }

    let final_ranking = sorted_reps
        .into_iter()
        .map(|rep| {
            let mut group = groups.remove(rep).unwrap_or_default();
            group.sort(); // Sort within ties for deterministic output
            group
        })
        .collect();

    Ok(final_ranking)
}

/// Pretty-prints the result of a function ranking.
///
/// # Arguments
/// * `title` - A title for the report to provide context.
/// * `ranking_result` - The result from the `rank_functions` call.
pub fn pretty_print_ranking(title: &str, ranking_result: &Result<Vec<Vec<String>>, String>) {
    eprintln!("--- {} ---", title);

    match ranking_result {
        // The ranking was successful
        Ok(ranking) => {
            if ranking.is_empty() {
                eprintln!("No functions were provided to rank.");
                eprintln!("-------------------------------------------------");
                return;
            }

            eprintln!("Performance Ranking (Fastest to Slowest)");
            eprintln!("----------------------------------------");

            for (i, rank_group) in ranking.iter().enumerate() {
                let rank = i + 1; // Use 1-based ranking for display

                // Add a newline for spacing between ranks, but not before the first one
                if i > 0 {
                    eprintln!();
                }

                // Check if this rank is a tie
                if rank_group.len() > 1 {
                    eprintln!("Rank {}: (Tied)", rank);
                } else {
                    eprintln!("Rank {}:", rank);
                }

                // Print each function ID in the group, indented for clarity
                for function_id in rank_group {
                    eprintln!("  - {}", function_id);
                }
            }
            eprintln!("-------------------------------------------------");
        }
        // An error occurred (e.g., a cycle was detected)
        Err(error_message) => {
            eprintln!("\n[ERROR] Could not generate ranking:");
            eprintln!("  Reason: {}", error_message);
            eprintln!("-------------------------------------------------");
        }
    }
    eprintln!(); // Add a final newline for spacing
}

// Rank fastest → slowest, returning groups of ties
pub fn rank_fastest(reports: &[ComparisonReportRanking]) -> Vec<Vec<String>> {
    // 0. Map every seen function-ID to a contiguous index
    let mut id_to_idx: HashMap<String, usize> = HashMap::new();
    let mut next_idx = 0usize;

    let mut intern = |id: &String, map: &mut HashMap<String, usize>, counter: &mut usize| {
        map.entry(id.clone()).or_insert_with(|| {
            let i = *counter;
            *counter += 1;
            i
        });
    };

    for r in reports {
        intern(&r.function_id_new, &mut id_to_idx, &mut next_idx);
        intern(&r.function_id_old, &mut id_to_idx, &mut next_idx);
    }

    // 1. Union-Find for ties (NonSignificant)
    let mut uf = QuickUnionUf::<UnionByRank>::new(next_idx);
    // let mut uf: QuickUnionUf<UnionByRank> = QuickUnionUf::new(next_idx);
    for r in reports {
        if matches!(r.result, ComparisonReportRankingResult::NonSignificant) {
            let a = id_to_idx[&r.function_id_new];
            let b = id_to_idx[&r.function_id_old];
            uf.union(a, b); // returns bool ↔ did_merge, we don't need it
        }
    }

    // 2. Individual win/lose scores
    let mut score = vec![0i32; next_idx];
    for r in reports {
        let (a, b) = (id_to_idx[&r.function_id_new], id_to_idx[&r.function_id_old]);
        match r.result {
            ComparisonReportRankingResult::Improved => {
                score[a] += 1;
                score[b] -= 1;
            }
            ComparisonReportRankingResult::Regressed => {
                score[a] -= 1;
                score[b] += 1;
            }
            ComparisonReportRankingResult::NonSignificant => {} // tie ⇒ no score change
        }
    }

    // 3. Aggregate scores per equivalence class (Union-Find root)
    let mut class_score: HashMap<usize, i32> = HashMap::new();
    for idx in 0..next_idx {
        let root = uf.find(idx);
        *class_score.entry(root).or_default() += score[idx];
    }

    // 4. Collect members per class
    let mut class_members: HashMap<usize, Vec<String>> = HashMap::new();
    for (id, &idx) in &id_to_idx {
        let root = uf.find(idx);
        class_members.entry(root).or_default().push(id.clone());
    }

    // 5. Sort classes by score (descending) and return member lists
    let mut ranked: Vec<(i32, Vec<String>)> = class_members
        .into_iter()
        .map(|(root, members)| (class_score[&root], members))
        .collect();

    ranked.sort_by(|a, b| b.0.cmp(&a.0)); // highest score first
    ranked.into_iter().map(|(_, v)| v).collect()
}

pub fn pretty_print_ranking2(ranking: &[Vec<String>]) {
    if ranking.is_empty() {
        eprintln!("(no data)");
        return;
    }

    let pad = ranking.len().to_string().len(); // width for rank numbers
    for (idx, group) in ranking.iter().enumerate() {
        // idx + 1  → human-friendly rank starting at 1
        eprintln!(
            " #{:>pad$}  {}{}",
            idx + 1,
            group.join(", "),
            match idx {
                0 => "          (fastest)",
                i if i + 1 == ranking.len() => "          (slowest)",
                _ => "",
            },
            pad = pad
        );
    }
}

pub struct ComparisonData {
    pub p_value: f64,
    pub t_distribution: Distribution<f64>,
    pub t_value: f64,
    pub relative_estimates: ChangeEstimates,
    pub relative_distributions: ChangeDistributions,
    pub significance_threshold: f64,
    pub noise_threshold: f64,
    pub base_iter_counts: Vec<f64>,
    pub base_sample_times: Vec<f64>,
    pub base_avg_times: Vec<f64>,
    pub base_estimates: Estimates,
}

pub struct MeasurementData<'a> {
    pub data: Data<'a, f64, f64>,
    pub avg_times: LabeledSample<'a, f64>,
    pub absolute_estimates: Estimates,
    pub distributions: Distributions,
    pub comparison: Option<ComparisonData>,
    pub throughput: Option<Throughput>,
}
impl MeasurementData<'_> {
    pub fn iter_counts(&self) -> &Sample<f64> {
        self.data.x()
    }

    pub fn sample_times(&self) -> &Sample<f64> {
        self.data.y()
    }
}

#[derive(Debug, Clone)]
pub struct OwnedMeasurementData {
    pub iter_counts: Vec<f64>,
    pub sample_times: Vec<f64>,
    pub avg_times: Vec<f64>,
    pub absolute_estimates: Estimates,
    pub distributions: Distributions,
    pub throughput: Option<Throughput>,
}

impl From<&MeasurementData<'_>> for OwnedMeasurementData {
    fn from(meas: &MeasurementData<'_>) -> Self {
        Self {
            iter_counts: meas.iter_counts().to_vec(),
            sample_times: meas.sample_times().to_vec(),
            avg_times: meas.avg_times.to_vec(),
            absolute_estimates: meas.absolute_estimates.clone(),
            distributions: meas.distributions.clone(),
            throughput: meas.throughput.clone(),
        }
    }
}

impl OwnedMeasurementData {
    pub fn iter_counts(&self) -> &Sample<f64> {
        Sample::new(&self.iter_counts)
    }

    pub fn sample_times(&self) -> &Sample<f64> {
        Sample::new(&self.sample_times)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ValueType {
    Bytes,
    Elements,
    Value,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BenchmarkId {
    pub group_id: String,
    pub function_id: Option<String>,
    pub value_str: Option<String>,
    pub throughput: Option<Throughput>,
    full_id: String,
    directory_name: PathBuf,
    title: String,
}

fn truncate_to_character_boundary(s: &mut String, max_len: usize) {
    let mut boundary = cmp::min(max_len, s.len());
    while !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    s.truncate(boundary);
}

pub fn make_filename_safe(string: &str) -> String {
    let mut string = string.replace(
        &['?', '"', '/', '\\', '*', '<', '>', ':', '|', '^'][..],
        "_",
    );

    // Truncate to last character boundary before max length...
    truncate_to_character_boundary(&mut string, MAX_DIRECTORY_NAME_LEN);

    if cfg!(target_os = "windows") {
        {
            string = string
                // On Windows, spaces in the end of the filename are ignored and will be trimmed.
                //
                // Without trimming ourselves, creating a directory `dir ` will silently create
                // `dir` instead, but then operations on files like `dir /file` will fail.
                //
                // Also note that it's important to do this *after* trimming to MAX_DIRECTORY_NAME_LEN,
                // otherwise it can trim again to a name with a trailing space.
                .trim_end()
                // On Windows, file names are not case-sensitive, so lowercase everything.
                .to_lowercase();
        }
    }

    string
}

impl BenchmarkId {
    pub fn new(
        group_id: String,
        function_id: Option<String>,
        value_str: Option<String>,
        throughput: Option<Throughput>,
    ) -> BenchmarkId {
        let full_id = match (&function_id, &value_str) {
            (Some(ref func), Some(ref val)) => format!("{}/{}/{}", group_id, func, val),
            (Some(ref func), &None) => format!("{}/{}", group_id, func),
            (&None, Some(ref val)) => format!("{}/{}", group_id, val),
            (&None, &None) => group_id.clone(),
        };

        let mut title = full_id.clone();
        truncate_to_character_boundary(&mut title, MAX_TITLE_LEN);
        if title != full_id {
            title.push_str("...");
        }

        let mut directory_name = PathBuf::from(make_filename_safe(&group_id));
        if let Some(func) = &function_id {
            directory_name.push(make_filename_safe(func));
        }
        if let Some(val) = &value_str {
            directory_name.push(make_filename_safe(val));
        }

        BenchmarkId {
            group_id,
            function_id,
            value_str,
            throughput,
            full_id,
            directory_name,
            title,
        }
    }

    pub fn as_title(&self) -> &str {
        &self.title
    }

    pub fn as_directory_name(&self) -> &Path {
        &self.directory_name
    }

    pub fn as_number(&self) -> Option<f64> {
        match self.throughput {
            Some(Throughput::Bytes(n))
            | Some(Throughput::BytesDecimal(n))
            | Some(Throughput::Elements(n)) => Some(n as f64),
            None => self
                .value_str
                .as_ref()
                .and_then(|string| string.parse::<f64>().ok()),
        }
    }

    pub fn value_type(&self) -> Option<ValueType> {
        match self.throughput {
            Some(Throughput::Bytes(_)) | Some(Throughput::BytesDecimal(_)) => {
                Some(ValueType::Bytes)
            }
            Some(Throughput::Elements(_)) => Some(ValueType::Elements),
            None => self
                .value_str
                .as_ref()
                .and_then(|string| string.parse::<f64>().ok())
                .map(|_| ValueType::Value),
        }
    }

    pub fn ensure_directory_name_unique(&mut self, existing_directories: &HashSet<PathBuf>) {
        if !existing_directories.contains(self.as_directory_name()) {
            return;
        }

        let mut counter = 2;
        loop {
            let mut file_name = self.as_directory_name().file_name().unwrap().to_os_string();
            file_name.push(format!("_{}", counter));
            let new_dir_name = self.as_directory_name().with_file_name(file_name);

            if !existing_directories.contains(&new_dir_name) {
                self.directory_name = new_dir_name;
                return;
            }
            counter += 1;
        }
    }

    pub fn ensure_title_unique(&mut self, existing_titles: &HashSet<String>) {
        if !existing_titles.contains(self.as_title()) {
            return;
        }

        let mut counter = 2;
        loop {
            let new_title = format!("{} #{}", self.as_title(), counter);
            if !existing_titles.contains(&new_title) {
                self.title = new_title;
                return;
            }
            counter += 1;
        }
    }
}
impl fmt::Display for BenchmarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_title())
    }
}
impl fmt::Debug for BenchmarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn format_opt(opt: &Option<String>) -> String {
            match *opt {
                Some(ref string) => format!("\"{}\"", string),
                None => "None".to_owned(),
            }
        }

        write!(
            f,
            "BenchmarkId {{ group_id: \"{}\", function_id: {}, value_str: {}, throughput: {:?} }}",
            self.group_id,
            format_opt(&self.function_id),
            format_opt(&self.value_str),
            self.throughput,
        )
    }
}

pub struct ReportContext {
    pub output_directory: PathBuf,
    pub plot_config: PlotConfiguration,
}
impl ReportContext {
    pub fn report_path<P: AsRef<Path> + ?Sized>(&self, id: &BenchmarkId, file_name: &P) -> PathBuf {
        path!(
            self.output_directory.clone(),
            id.as_directory_name(),
            file_name
        )
    }
}

pub trait Report {
    fn benchmark_start(&self, _id: &BenchmarkId, _context: &ReportContext) {}
    fn warmup(&self, _id: &BenchmarkId, _context: &ReportContext, _warmup_ns: f64) {}
    fn analysis(&self, _id: &BenchmarkId, _context: &ReportContext) {}
    fn measurement_start(
        &self,
        _id: &BenchmarkId,
        _context: &ReportContext,
        _sample_count: u64,
        _estimate_ns: f64,
        _iter_count: u64,
    ) {
    }
    fn measurement_complete(
        &self,
        _id: &BenchmarkId,
        _context: &ReportContext,
        _measurements: &MeasurementData<'_>,
        _formatter: &ValueFormatter,
    ) {
    }
    fn summarize(
        &self,
        _context: &ReportContext,
        _group_id: &str,
        _benchmark_group: &BenchmarkGroup,
        _formatter: &ValueFormatter,
    ) {
    }
    fn final_summary(&self, _context: &ReportContext, _model: &Model) {}
    fn group_separator(&self) {}
    fn history(
        &self,
        _context: &ReportContext,
        _id: &BenchmarkId,
        _history: &[SavedStatistics],
        _formatter: &ValueFormatter,
    ) {
    }

    fn show_intra_group_comparison(
        &self,
        group_id: &str,
        comparisons: &Vec<ComparisonReport>,
        report_context: &ReportContext,
        formatter: &ValueFormatter,
    ) {
    }
}

pub enum CliReports {
    Cli(CliReport),
    CliIntraGroup(CliReportIntraGroup),
}

impl Report for CliReports {
    fn benchmark_start(&self, id: &BenchmarkId, context: &ReportContext) {
        match self {
            CliReports::Cli(report) => report.benchmark_start(id, context),
            CliReports::CliIntraGroup(report) => report.benchmark_start(id, context),
        }
    }

    fn warmup(&self, id: &BenchmarkId, context: &ReportContext, warmup_ns: f64) {
        match self {
            CliReports::Cli(report) => report.warmup(id, context, warmup_ns),
            CliReports::CliIntraGroup(report) => report.warmup(id, context, warmup_ns),
        }
    }

    fn analysis(&self, id: &BenchmarkId, context: &ReportContext) {
        match self {
            CliReports::Cli(report) => report.analysis(id, context),
            CliReports::CliIntraGroup(report) => report.analysis(id, context),
        }
    }

    fn measurement_start(
        &self,
        id: &BenchmarkId,
        context: &ReportContext,
        sample_count: u64,
        estimate_ns: f64,
        iter_count: u64,
    ) {
        match self {
            CliReports::Cli(report) => {
                report.measurement_start(id, context, sample_count, estimate_ns, iter_count);
            }
            CliReports::CliIntraGroup(report) => {
                report.measurement_start(id, context, sample_count, estimate_ns, iter_count);
            }
        }
    }

    fn measurement_complete(
        &self,
        id: &BenchmarkId,
        context: &ReportContext,
        measurements: &MeasurementData<'_>,
        formatter: &ValueFormatter,
    ) {
        match self {
            CliReports::Cli(report) => {
                report.measurement_complete(id, context, measurements, formatter);
            }
            CliReports::CliIntraGroup(report) => {
                report.measurement_complete(id, context, measurements, formatter);
            }
        }
    }

    fn group_separator(&self) {
        match self {
            CliReports::Cli(report) => report.group_separator(),
            CliReports::CliIntraGroup(report) => report.group_separator(),
        }
    }

    fn show_intra_group_comparison(
        &self,
        group_id: &str,
        comparisons: &Vec<ComparisonReport>,
        report_context: &ReportContext,
        formatter: &ValueFormatter,
    ) {
        match self {
            CliReports::Cli(_) => {}
            CliReports::CliIntraGroup(report) => {
                report.show_intra_group_comparison(
                    group_id,
                    comparisons,
                    report_context,
                    formatter,
                );
            }
        }
    }
}

pub struct Reports<'a> {
    reports: Vec<&'a dyn Report>,
}
impl<'a> Reports<'a> {
    pub fn new(reports: Vec<&'a dyn Report>) -> Reports<'a> {
        Reports { reports }
    }
}
impl Report for Reports<'_> {
    fn benchmark_start(&self, id: &BenchmarkId, context: &ReportContext) {
        for report in &self.reports {
            report.benchmark_start(id, context);
        }
    }

    fn warmup(&self, id: &BenchmarkId, context: &ReportContext, warmup_ns: f64) {
        for report in &self.reports {
            report.warmup(id, context, warmup_ns);
        }
    }

    fn analysis(&self, id: &BenchmarkId, context: &ReportContext) {
        for report in &self.reports {
            report.analysis(id, context);
        }
    }

    fn measurement_start(
        &self,
        id: &BenchmarkId,
        context: &ReportContext,
        sample_count: u64,
        estimate_ns: f64,
        iter_count: u64,
    ) {
        for report in &self.reports {
            report.measurement_start(id, context, sample_count, estimate_ns, iter_count);
        }
    }

    fn measurement_complete(
        &self,
        id: &BenchmarkId,
        context: &ReportContext,
        measurements: &MeasurementData<'_>,
        formatter: &ValueFormatter,
    ) {
        for report in &self.reports {
            report.measurement_complete(id, context, measurements, formatter);
        }
    }

    fn summarize(
        &self,
        context: &ReportContext,
        group_id: &str,
        benchmark_group: &BenchmarkGroup,
        formatter: &ValueFormatter,
    ) {
        for report in &self.reports {
            report.summarize(context, group_id, benchmark_group, formatter);
        }
    }

    fn final_summary(&self, context: &ReportContext, model: &Model) {
        for report in &self.reports {
            report.final_summary(context, model);
        }
    }

    fn group_separator(&self) {
        for report in &self.reports {
            report.group_separator();
        }
    }

    fn history(
        &self,
        context: &ReportContext,
        id: &BenchmarkId,
        history: &[SavedStatistics],
        formatter: &ValueFormatter,
    ) {
        for report in &self.reports {
            report.history(context, id, history, formatter);
        }
    }

    fn show_intra_group_comparison(
        &self,
        group_id: &str,
        comparisons: &Vec<ComparisonReport>,
        report_context: &ReportContext,
        formatter: &ValueFormatter,
    ) {
        for report in &self.reports {
            report.show_intra_group_comparison(group_id, comparisons, report_context, formatter);
        }
    }
}

pub struct CliReport {
    pub enable_text_overwrite: bool,
    pub enable_text_coloring: bool,
    pub verbose: bool,
    pub show_differences: bool,

    last_line_len: Cell<usize>,
}
impl CliReport {
    pub fn new(
        enable_text_overwrite: bool,
        enable_text_coloring: bool,
        show_differences: bool,
        verbose: bool,
    ) -> CliReport {
        CliReport {
            enable_text_overwrite,
            enable_text_coloring,
            show_differences,
            verbose,

            last_line_len: Cell::new(0),
        }
    }

    fn text_overwrite(&self) {
        if self.enable_text_overwrite {
            eprint!("\r");
            for _ in 0..self.last_line_len.get() {
                eprint!(" ");
            }
            eprint!("\r");
        }
    }

    //Passing a String is the common case here.
    #[allow(clippy::needless_pass_by_value)]
    fn print_overwritable(&self, s: String) {
        if self.enable_text_overwrite {
            self.last_line_len.set(s.len());
            eprint!("{}", s);
            stderr().flush().unwrap();
        } else {
            eprintln!("{}", s);
        }
    }

    fn green(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[32m{}\x1B[39m", s)
        } else {
            s
        }
    }

    fn yellow(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[33m{}\x1B[39m", s)
        } else {
            s
        }
    }

    fn red(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[31m{}\x1B[39m", s)
        } else {
            s
        }
    }

    fn bold(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[1m{}\x1B[22m", s)
        } else {
            s
        }
    }

    fn faint(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[2m{}\x1B[22m", s)
        } else {
            s
        }
    }

    pub fn outliers(&self, sample: &LabeledSample<'_, f64>) {
        let (los, lom, _, him, his) = sample.count();
        let noutliers = los + lom + him + his;
        let sample_size = sample.len();

        if noutliers == 0 {
            return;
        }

        let percent = |n: usize| 100. * n as f64 / sample_size as f64;

        eprintln!(
            "{}",
            self.yellow(format!(
                "Found {} outliers among {} measurements ({:.2}%)",
                noutliers,
                sample_size,
                percent(noutliers)
            ))
        );

        let print = |n, label| {
            if n != 0 {
                eprintln!("  {} ({:.2}%) {}", n, percent(n), label);
            }
        };

        print(los, "low severe");
        print(lom, "low mild");
        print(him, "high mild");
        print(his, "high severe");
    }
}
impl Report for CliReport {
    fn benchmark_start(&self, id: &BenchmarkId, _: &ReportContext) {
        self.print_overwritable(format!("Benchmarking {}", id));
    }

    fn warmup(&self, id: &BenchmarkId, _: &ReportContext, warmup_ns: f64) {
        self.text_overwrite();
        self.print_overwritable(format!(
            "Benchmarking {}: Warming up for {}",
            id,
            format::time(warmup_ns)
        ));
    }

    fn analysis(&self, id: &BenchmarkId, _: &ReportContext) {
        self.text_overwrite();
        self.print_overwritable(format!("Benchmarking {}: Analyzing", id));
    }

    fn measurement_start(
        &self,
        id: &BenchmarkId,
        _: &ReportContext,
        sample_count: u64,
        estimate_ns: f64,
        iter_count: u64,
    ) {
        self.text_overwrite();
        let iter_string = if self.verbose {
            format!("{} iterations", iter_count)
        } else {
            format::iter_count(iter_count)
        };

        self.print_overwritable(format!(
            "Benchmarking {}: Collecting {} samples in estimated {} ({})",
            id,
            sample_count,
            format::time(estimate_ns),
            iter_string
        ));
    }

    fn measurement_complete(
        &self,
        id: &BenchmarkId,
        _: &ReportContext,
        meas: &MeasurementData<'_>,
        formatter: &ValueFormatter,
    ) {
        self.text_overwrite();

        let typical_estimate = meas.absolute_estimates.typical();

        {
            let mut id = id.as_title().to_owned();

            if id.len() > 23 {
                eprintln!("{}", self.green(id.clone()));
                id.clear();
            }
            let id_len = id.len();

            eprintln!(
                "{}{}time:   [{} {} {}]",
                self.green(id),
                " ".repeat(24 - id_len),
                self.faint(
                    formatter.format_value(typical_estimate.confidence_interval.lower_bound)
                ),
                self.bold(formatter.format_value(typical_estimate.point_estimate)),
                self.faint(
                    formatter.format_value(typical_estimate.confidence_interval.upper_bound)
                )
            );
        }

        if let Some(ref throughput) = meas.throughput {
            eprintln!(
                "{}thrpt:  [{} {} {}]",
                " ".repeat(24),
                self.faint(formatter.format_throughput(
                    throughput,
                    typical_estimate.confidence_interval.upper_bound
                )),
                self.bold(formatter.format_throughput(throughput, typical_estimate.point_estimate)),
                self.faint(formatter.format_throughput(
                    throughput,
                    typical_estimate.confidence_interval.lower_bound
                )),
            )
        }

        if self.show_differences {
            if let Some(ref comp) = meas.comparison {
                let different_mean = comp.p_value < comp.significance_threshold;
                let mean_est = &comp.relative_estimates.mean;
                let point_estimate = mean_est.point_estimate;
                let mut point_estimate_str = format::change(point_estimate, true);
                // The change in throughput is related to the change in timing. Reducing the timing by
                // 50% increases the througput by 100%.
                let to_thrpt_estimate = |ratio: f64| 1.0 / (1.0 + ratio) - 1.0;
                let mut thrpt_point_estimate_str =
                    format::change(to_thrpt_estimate(point_estimate), true);
                let explanation_str: String;

                if !different_mean {
                    explanation_str = "No change in performance detected.".to_owned();
                } else {
                    let comparison = compare_to_threshold(mean_est, comp.noise_threshold);
                    match comparison {
                        ComparisonResult::Improved => {
                            point_estimate_str = self.green(self.bold(point_estimate_str));
                            thrpt_point_estimate_str =
                                self.green(self.bold(thrpt_point_estimate_str));
                            explanation_str =
                                format!("Performance has {}.", self.green("improved".to_owned()));
                        }
                        ComparisonResult::Regressed => {
                            point_estimate_str = self.red(self.bold(point_estimate_str));
                            thrpt_point_estimate_str =
                                self.red(self.bold(thrpt_point_estimate_str));
                            explanation_str =
                                format!("Performance has {}.", self.red("regressed".to_owned()));
                        }
                        ComparisonResult::NonSignificant => {
                            explanation_str = "Change within noise threshold.".to_owned();
                        }
                    }
                }

                if meas.throughput.is_some() {
                    eprintln!("{}change:", " ".repeat(17));

                    eprintln!(
                        "{}time:   [{} {} {}] (p = {:.2} {} {:.2})",
                        " ".repeat(24),
                        self.faint(format::change(
                            mean_est.confidence_interval.lower_bound,
                            true
                        )),
                        point_estimate_str,
                        self.faint(format::change(
                            mean_est.confidence_interval.upper_bound,
                            true
                        )),
                        comp.p_value,
                        if different_mean { "<" } else { ">" },
                        comp.significance_threshold
                    );
                    eprintln!(
                        "{}thrpt:  [{} {} {}]",
                        " ".repeat(24),
                        self.faint(format::change(
                            to_thrpt_estimate(mean_est.confidence_interval.upper_bound),
                            true
                        )),
                        thrpt_point_estimate_str,
                        self.faint(format::change(
                            to_thrpt_estimate(mean_est.confidence_interval.lower_bound),
                            true
                        )),
                    );
                } else {
                    eprintln!(
                        "{}change: [{} {} {}] (p = {:.2} {} {:.2})",
                        " ".repeat(24),
                        self.faint(format::change(
                            mean_est.confidence_interval.lower_bound,
                            true
                        )),
                        point_estimate_str,
                        self.faint(format::change(
                            mean_est.confidence_interval.upper_bound,
                            true
                        )),
                        comp.p_value,
                        if different_mean { "<" } else { ">" },
                        comp.significance_threshold
                    );
                }

                eprintln!("{}{}", " ".repeat(24), explanation_str);
            }
        }

        if self.verbose {
            self.outliers(&meas.avg_times);

            let format_short_estimate = |estimate: &Estimate| -> String {
                format!(
                    "[{} {}]",
                    formatter.format_value(estimate.confidence_interval.lower_bound),
                    formatter.format_value(estimate.confidence_interval.upper_bound)
                )
            };

            let data = &meas.data;
            if let Some(slope_estimate) = meas.absolute_estimates.slope.as_ref() {
                eprintln!(
                    "{:<7}{} {:<15}[{:0.7} {:0.7}]",
                    "slope",
                    format_short_estimate(slope_estimate),
                    "R^2",
                    Slope(slope_estimate.confidence_interval.lower_bound).r_squared(data),
                    Slope(slope_estimate.confidence_interval.upper_bound).r_squared(data),
                );
            }
            eprintln!(
                "{:<7}{} {:<15}{}",
                "mean",
                format_short_estimate(&meas.absolute_estimates.mean),
                "std. dev.",
                format_short_estimate(&meas.absolute_estimates.std_dev),
            );
            eprintln!(
                "{:<7}{} {:<15}{}",
                "median",
                format_short_estimate(&meas.absolute_estimates.median),
                "med. abs. dev.",
                format_short_estimate(&meas.absolute_estimates.median_abs_dev),
            );
        }
    }

    fn group_separator(&self) {
        eprintln!();
    }
}

pub struct CliReportIntraGroup {
    pub enable_text_overwrite: bool,
    pub enable_text_coloring: bool,
    pub verbose: bool,
    pub show_differences: bool,

    last_line_len: Cell<usize>,
}

impl CliReportIntraGroup {
    pub fn new(
        enable_text_overwrite: bool,
        enable_text_coloring: bool,
        show_differences: bool,
        verbose: bool,
    ) -> CliReportIntraGroup {
        CliReportIntraGroup {
            enable_text_overwrite,
            enable_text_coloring,
            show_differences,
            verbose,

            last_line_len: Cell::new(0),
        }
    }

    fn text_overwrite(&self) {
        if self.enable_text_overwrite {
            eprint!("\r");
            for _ in 0..self.last_line_len.get() {
                eprint!(" ");
            }
            eprint!("\r");
        }
    }

    //Passing a String is the common case here.
    #[allow(clippy::needless_pass_by_value)]
    fn print_overwritable(&self, s: String) {
        if self.enable_text_overwrite {
            self.last_line_len.set(s.len());
            eprint!("{}", s);
            stderr().flush().unwrap();
        } else {
            eprintln!("{}", s);
            eprintln!("CliReportIntraGroup - print_overwritable");
        }
    }

    fn green(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[32m{}\x1B[39m", s)
        } else {
            s
        }
    }

    fn yellow(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[33m{}\x1B[39m", s)
        } else {
            s
        }
    }

    fn red(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[31m{}\x1B[39m", s)
        } else {
            s
        }
    }

    fn bold(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[1m{}\x1B[22m", s)
        } else {
            s
        }
    }

    fn faint(&self, s: String) -> String {
        if self.enable_text_coloring {
            format!("\x1B[2m{}\x1B[22m", s)
        } else {
            s
        }
    }

    pub fn outliers(&self, sample: &LabeledSample<'_, f64>) {
        let (los, lom, _, him, his) = sample.count();
        let noutliers = los + lom + him + his;
        let sample_size = sample.len();

        if noutliers == 0 {
            return;
        }

        let percent = |n: usize| 100. * n as f64 / sample_size as f64;

        eprintln!(
            "{}",
            self.yellow(format!(
                "Found {} outliers among {} measurements ({:.2}%)",
                noutliers,
                sample_size,
                percent(noutliers)
            ))
        );

        let print = |n, label| {
            if n != 0 {
                eprintln!("  {} ({:.2}%) {}", n, percent(n), label);
            }
        };

        print(los, "low severe");
        print(lom, "low mild");
        print(him, "high mild");
        print(his, "high severe");
    }
}
impl Report for CliReportIntraGroup {
    fn benchmark_start(&self, id: &BenchmarkId, _: &ReportContext) {
        self.print_overwritable(format!("Benchmarking {}", id));
    }

    fn warmup(&self, id: &BenchmarkId, _: &ReportContext, warmup_ns: f64) {
        self.text_overwrite();
        self.print_overwritable(format!(
            "Benchmarking {}: Warming up for {}",
            id,
            format::time(warmup_ns)
        ));
    }

    fn analysis(&self, id: &BenchmarkId, _: &ReportContext) {
        self.text_overwrite();
        self.print_overwritable(format!("Benchmarking {}: Analyzing", id));
    }

    fn measurement_start(
        &self,
        id: &BenchmarkId,
        _: &ReportContext,
        sample_count: u64,
        estimate_ns: f64,
        iter_count: u64,
    ) {
        self.text_overwrite();
        let iter_string = if self.verbose {
            format!("{} iterations", iter_count)
        } else {
            format::iter_count(iter_count)
        };

        self.print_overwritable(format!(
            "Benchmarking {}: Collecting {} samples in estimated {} ({})",
            id,
            sample_count,
            format::time(estimate_ns),
            iter_string
        ));
    }

    fn measurement_complete(
        &self,
        id: &BenchmarkId,
        _: &ReportContext,
        meas: &MeasurementData<'_>,
        formatter: &ValueFormatter,
    ) {
        self.text_overwrite();

        let typical_estimate = meas.absolute_estimates.typical();

        {
            let mut id = id.as_title().to_owned();

            if id.len() > 23 {
                eprintln!("{}", self.green(id.clone()));
                id.clear();
            }
            let id_len = id.len();

            eprintln!("Intra-Group Comparison");

            eprintln!(
                "{}{}time:   [{} {} {}]",
                self.green(id),
                " ".repeat(24 - id_len),
                self.faint(
                    formatter.format_value(typical_estimate.confidence_interval.lower_bound)
                ),
                self.bold(formatter.format_value(typical_estimate.point_estimate)),
                self.faint(
                    formatter.format_value(typical_estimate.confidence_interval.upper_bound)
                )
            );
        }

        if let Some(ref throughput) = meas.throughput {
            eprintln!(
                "{}thrpt:  [{} {} {}]",
                " ".repeat(24),
                self.faint(formatter.format_throughput(
                    throughput,
                    typical_estimate.confidence_interval.upper_bound
                )),
                self.bold(formatter.format_throughput(throughput, typical_estimate.point_estimate)),
                self.faint(formatter.format_throughput(
                    throughput,
                    typical_estimate.confidence_interval.lower_bound
                )),
            )
        }

        // if self.show_differences {
        //     if let Some(ref comp) = meas.comparison {
        //         let different_mean = comp.p_value < comp.significance_threshold;
        //         let mean_est = &comp.relative_estimates.mean;
        //         let point_estimate = mean_est.point_estimate;
        //         let mut point_estimate_str = format::change(point_estimate, true);
        //         // The change in throughput is related to the change in timing. Reducing the timing by
        //         // 50% increases the througput by 100%.
        //         let to_thrpt_estimate = |ratio: f64| 1.0 / (1.0 + ratio) - 1.0;
        //         let mut thrpt_point_estimate_str =
        //             format::change(to_thrpt_estimate(point_estimate), true);
        //         let explanation_str: String;

        //         if !different_mean {
        //             explanation_str = "No change in performance detected.".to_owned();
        //         } else {
        //             let comparison = compare_to_threshold(mean_est, comp.noise_threshold);
        //             match comparison {
        //                 ComparisonResult::Improved => {
        //                     point_estimate_str = self.green(self.bold(point_estimate_str));
        //                     thrpt_point_estimate_str =
        //                         self.green(self.bold(thrpt_point_estimate_str));
        //                     explanation_str =
        //                         format!("Performance has {}.", self.green("improved".to_owned()));
        //                 }
        //                 ComparisonResult::Regressed => {
        //                     point_estimate_str = self.red(self.bold(point_estimate_str));
        //                     thrpt_point_estimate_str =
        //                         self.red(self.bold(thrpt_point_estimate_str));
        //                     explanation_str =
        //                         format!("Performance has {}.", self.red("regressed".to_owned()));
        //                 }
        //                 ComparisonResult::NonSignificant => {
        //                     explanation_str = "Change within noise threshold.".to_owned();
        //                 }
        //             }
        //         }

        //         if meas.throughput.is_some() {
        //             eprintln!("{}change:", " ".repeat(17));

        //             eprintln!(
        //                 "{}time:   [{} {} {}] (p = {:.2} {} {:.2})",
        //                 " ".repeat(24),
        //                 self.faint(format::change(
        //                     mean_est.confidence_interval.lower_bound,
        //                     true
        //                 )),
        //                 point_estimate_str,
        //                 self.faint(format::change(
        //                     mean_est.confidence_interval.upper_bound,
        //                     true
        //                 )),
        //                 comp.p_value,
        //                 if different_mean { "<" } else { ">" },
        //                 comp.significance_threshold
        //             );
        //             eprintln!(
        //                 "{}thrpt:  [{} {} {}]",
        //                 " ".repeat(24),
        //                 self.faint(format::change(
        //                     to_thrpt_estimate(mean_est.confidence_interval.upper_bound),
        //                     true
        //                 )),
        //                 thrpt_point_estimate_str,
        //                 self.faint(format::change(
        //                     to_thrpt_estimate(mean_est.confidence_interval.lower_bound),
        //                     true
        //                 )),
        //             );
        //         } else {
        //             eprintln!(
        //                 "{}change: [{} {} {}] (p = {:.2} {} {:.2})",
        //                 " ".repeat(24),
        //                 self.faint(format::change(
        //                     mean_est.confidence_interval.lower_bound,
        //                     true
        //                 )),
        //                 point_estimate_str,
        //                 self.faint(format::change(
        //                     mean_est.confidence_interval.upper_bound,
        //                     true
        //                 )),
        //                 comp.p_value,
        //                 if different_mean { "<" } else { ">" },
        //                 comp.significance_threshold
        //             );
        //         }

        //         eprintln!("{}{}", " ".repeat(24), explanation_str);
        //     }
        // }

        if self.verbose {
            self.outliers(&meas.avg_times);

            let format_short_estimate = |estimate: &Estimate| -> String {
                format!(
                    "[{} {}]",
                    formatter.format_value(estimate.confidence_interval.lower_bound),
                    formatter.format_value(estimate.confidence_interval.upper_bound)
                )
            };

            let data = &meas.data;
            if let Some(slope_estimate) = meas.absolute_estimates.slope.as_ref() {
                eprintln!(
                    "{:<7}{} {:<15}[{:0.7} {:0.7}]",
                    "slope",
                    format_short_estimate(slope_estimate),
                    "R^2",
                    Slope(slope_estimate.confidence_interval.lower_bound).r_squared(data),
                    Slope(slope_estimate.confidence_interval.upper_bound).r_squared(data),
                );
            }
            eprintln!(
                "{:<7}{} {:<15}{}",
                "mean",
                format_short_estimate(&meas.absolute_estimates.mean),
                "std. dev.",
                format_short_estimate(&meas.absolute_estimates.std_dev),
            );
            eprintln!(
                "{:<7}{} {:<15}{}",
                "median",
                format_short_estimate(&meas.absolute_estimates.median),
                "med. abs. dev.",
                format_short_estimate(&meas.absolute_estimates.median_abs_dev),
            );
        }
    }

    fn group_separator(&self) {
        eprintln!();
    }

    fn show_intra_group_comparison(
        &self,
        group_id: &str,
        comparisons: &Vec<ComparisonReport>,
        report_context: &ReportContext,
        formatter: &ValueFormatter,
    ) {
        self.text_overwrite();

        let mut comparison_report_results: Vec<ComparisonReportRanking> = Vec::new();
        let mut data: Vec<Vec<CellStruct>> = Vec::new();
        for comparison in comparisons {
            let mut rows: Vec<CellStruct> = Vec::new();

            let is_mean_different =
                comparison.comp.p_value < comparison.comp.significance_threshold;
            let mean_diff_est = &comparison.comp.relative_estimates.mean;
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
            let mean_diff_ci_lower_bound =
                mean_diff_est.confidence_interval.lower_bound * benchmark_old_mean;
            let mean_diff_ci_upper_bound =
                mean_diff_est.confidence_interval.upper_bound * benchmark_old_mean;
            let mean_diff_pct_str = format::change(mean_diff_point_estimate.abs(), false);
            let noise_threshold = comparison.comp.noise_threshold;
            let function_id_old_str = comparison.id_old.function_id.as_ref().unwrap().to_owned();
            let function_id_new_str = comparison.id_new.function_id.as_ref().unwrap().to_owned();
            let explanation_str: String;

            let mut mean_diff = format!("{:+.2} ns", mean_diff_point_estimate * benchmark_old_mean);
            // let mut function_id_old_color_str =
            // comparison.id_old.function_id.as_ref().unwrap().to_owned();
            // let mut function_id_new_color_str =
            // comparison.id_new.function_id.as_ref().unwrap().to_owned();
            let mut function_id_old_color_str = function_id_old_str.clone();
            let mut function_id_new_color_str = function_id_new_str.clone();
            let mut benchmark_old_mean_str = formatter.format_value(benchmark_old_mean);
            let mut benchmark_new_mean_str = formatter.format_value(benchmark_new_mean);

            if is_mean_different {
                let comparison = compare_to_threshold(mean_diff_est, noise_threshold);
                match comparison {
                    ComparisonResult::Improved => {
                        mean_diff = self.green(self.bold(mean_diff));
                        benchmark_new_mean_str = self.green(self.bold(benchmark_new_mean_str));
                        benchmark_old_mean_str = self.red(benchmark_old_mean_str);
                        function_id_new_color_str =
                            self.green(self.bold(function_id_new_color_str));
                        function_id_old_color_str = self.red(function_id_old_color_str);
                        explanation_str = format!(
                            "Performance has {}",
                            self.green(format!("improved {mean_diff_pct_str}"))
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
                            self.red(format!("regressed {mean_diff_pct_str}"))
                        );
                        comparison_report_results.push(ComparisonReportRanking {
                            function_id_new: function_id_new_str,
                            function_id_old: function_id_old_str,
                            result: ComparisonReportRankingResult::Regressed,
                        });
                    }
                    ComparisonResult::NonSignificant => {
                        explanation_str = format!(
                            "Changed {mean_diff_pct_str} within noise threshold of {noise_threshold:.2} ns",
                        );
                        comparison_report_results.push(ComparisonReportRanking {
                            function_id_new: function_id_new_str,
                            function_id_old: function_id_old_str,
                            result: ComparisonReportRankingResult::NonSignificant,
                        });
                    }
                }
            } else {
                explanation_str = "No change in performance detected".to_owned();
                comparison_report_results.push(ComparisonReportRanking {
                    function_id_new: function_id_new_str,
                    function_id_old: function_id_old_str,
                    // result: ComparisonReportResult::NoChange,
                    result: ComparisonReportRankingResult::NonSignificant,
                });
            }

            rows.push(
                format!(
                    "{} vs {}",
                    &function_id_old_color_str, &function_id_new_color_str
                )
                .cell()
                .justify(Justify::Center)
                .align(Align::Center),
            );
            rows.push(
                format!("{} vs {}", &benchmark_old_mean_str, &benchmark_new_mean_str)
                    .cell()
                    .justify(Justify::Center)
                    .align(Align::Center),
            );
            rows.push(
                format!(
                    "{} [{:+.2},{:+.2}] {} CI (p = {:.12} {} {:.3})",
                    &mean_diff,
                    mean_diff_ci_lower_bound,
                    mean_diff_ci_upper_bound,
                    format::change(mean_diff_est.confidence_interval.confidence_level, false),
                    &comparison.comp.p_value,
                    if is_mean_different { "<" } else { ">" },
                    &comparison.comp.significance_threshold
                )
                .cell()
                .justify(Justify::Center)
                .align(Align::Center),
            );
            rows.push(
                explanation_str
                    .cell()
                    .justify(Justify::Center)
                    .align(Align::Center),
            );
            data.push(rows);
        }
        let data_table = data
            .table()
            .title(vec![
                group_id
                    .cell()
                    .justify(Justify::Center)
                    .align(Align::Center)
                    .bold(true),
                "Latency (mean)"
                    .cell()
                    .justify(Justify::Center)
                    .align(Align::Center)
                    .bold(true),
                "Latency Change (mean)"
                    .cell()
                    .justify(Justify::Center)
                    .align(Align::Center)
                    .bold(true),
                "Result"
                    .cell()
                    .justify(Justify::Center)
                    .align(Align::Center)
                    .bold(true),
            ])
            .separator(
                Separator::builder()
                    .row(Some(HorizontalLine::new('├', '┤', '┼', '─')))
                    .title(Some(HorizontalLine::new('├', '┤', '┼', '─')))
                    .column(Some(VerticalLine::new('│')))
                    .build(),
            )
            .border(
                Border::builder()
                    // .top(HorizontalLine::new('╭', '╮', '┬', '─'))
                    .top(HorizontalLine::new('┌', '┐', '┬', '─'))
                    // .bottom(HorizontalLine::new('╰', '╯', '┴', '─'))
                    .bottom(HorizontalLine::new('└', '┘', '┴', '─'))
                    .left(VerticalLine::new('│'))
                    .right(VerticalLine::new('│'))
                    .build(),
            )
            .bold(true);

        let _ = print_stderr(data_table);

        let ranking1 = rank_functions(&comparison_report_results);
        pretty_print_ranking("ranking1", &ranking1);

        let ranking2 = rank_fastest(&comparison_report_results);

        pretty_print_ranking2(&ranking2);
        // for r in comparison_report_results {
        //     eprintln!("{r:?}");
        // }
    }
}

pub struct BencherReport;
impl Report for BencherReport {
    fn measurement_start(
        &self,
        id: &BenchmarkId,
        _context: &ReportContext,
        _sample_count: u64,
        _estimate_ns: f64,
        _iter_count: u64,
    ) {
        eprint!("test {} ... ", id);
    }

    fn measurement_complete(
        &self,
        _id: &BenchmarkId,
        _: &ReportContext,
        meas: &MeasurementData<'_>,
        formatter: &ValueFormatter,
    ) {
        let mut values = [
            meas.absolute_estimates.median.point_estimate,
            meas.absolute_estimates.std_dev.point_estimate,
        ];
        let unit = formatter.scale_for_machines(&mut values);

        eprintln!(
            "bench: {:>11} {}/iter (+/- {})",
            format::integer(values[0]),
            unit,
            format::integer(values[1])
        );
    }

    fn group_separator(&self) {
        eprintln!();
    }
}

pub enum ComparisonResult {
    Improved,
    Regressed,
    NonSignificant,
}

pub fn compare_to_threshold(estimate: &Estimate, noise: f64) -> ComparisonResult {
    let ci = &estimate.confidence_interval;
    let lb = ci.lower_bound;
    let ub = ci.upper_bound;

    if lb < -noise && ub < -noise {
        ComparisonResult::Improved
    } else if lb > noise && ub > noise {
        ComparisonResult::Regressed
    } else {
        ComparisonResult::NonSignificant
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_make_filename_safe_replaces_characters() {
        let input = "?/\\*\"";
        let safe = make_filename_safe(input);
        assert_eq!("_____", &safe);
    }

    #[test]
    fn test_make_filename_safe_truncates_long_strings() {
        let input = "this is a very long string. it is too long to be safe as a directory name, and so it needs to be truncated. what a long string this is.";
        let safe = make_filename_safe(input);
        assert!(input.len() > MAX_DIRECTORY_NAME_LEN);
        assert_eq!(&input[0..MAX_DIRECTORY_NAME_LEN], &safe);
    }

    #[test]
    fn test_make_filename_safe_respects_character_boundaries() {
        let input = "✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓";
        let safe = make_filename_safe(input);
        assert!(safe.len() < MAX_DIRECTORY_NAME_LEN);
    }

    #[test]
    fn test_benchmark_id_make_directory_name_unique() {
        let existing_id = BenchmarkId::new(
            "group".to_owned(),
            Some("function".to_owned()),
            Some("value".to_owned()),
            None,
        );
        let mut directories = HashSet::new();
        directories.insert(existing_id.as_directory_name().to_owned());

        let mut new_id = existing_id.clone();
        new_id.ensure_directory_name_unique(&directories);
        assert_eq!(
            "group/function/value_2",
            new_id.as_directory_name().to_str().unwrap()
        );
        directories.insert(new_id.as_directory_name().to_owned());

        new_id = existing_id.clone();
        new_id.ensure_directory_name_unique(&directories);
        assert_eq!(
            "group/function/value_3",
            new_id.as_directory_name().to_str().unwrap()
        );
        directories.insert(new_id.as_directory_name().to_owned());
    }
    #[test]
    fn test_benchmark_id_make_long_directory_name_unique() {
        let long_name = (0..MAX_DIRECTORY_NAME_LEN).map(|_| 'a').collect::<String>();
        let existing_id = BenchmarkId::new(long_name, None, None, None);
        let mut directories = HashSet::new();
        directories.insert(existing_id.as_directory_name().to_owned());

        let mut new_id = existing_id.clone();
        new_id.ensure_directory_name_unique(&directories);
        assert_ne!(existing_id.as_directory_name(), new_id.as_directory_name());
    }
}
