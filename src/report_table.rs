use tabled::{
    grid::config::ColoredConfig,
    grid::records::{ExactRecords, PeekableRecords, Records},
    settings::{style::Style, themes::BorderCorrection, Alignment, Format, TableOption},
    Table, Tabled,
};

pub struct ChangesData {
    pub group_id: String,
    pub changes_table_rows: Vec<ChangesTable>,
    pub ranking_table_rows: Vec<RankingTable>,
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

pub fn print_changes_table(group_id: &str, rows: &[ChangesTable]) {
    let mut table = Table::new(rows);

    table.modify((0, 0), Format::content(|_| group_id.to_string()));

    table
        // .with(Style::modern_rounded())
        .with(Style::modern())
        // .with(MergeDuplicatesVerticalFirst)
        // .with(BorderCorrection::span())
        .with(Alignment::center())
        .with(Alignment::center_vertical());

    eprintln!("{table}");
}

pub fn print_ranking_table(group_id: &str, rows: &[RankingTable]) {
    let mut table = Table::new(rows);

    table
        // .with(Style::modern_rounded())
        .with(Style::modern())
        .with(MergeDuplicatesVerticalFirst)
        .with(BorderCorrection::span())
        .with(Alignment::center())
        .with(Alignment::center_vertical());

    eprintln!("{table}");
}
