use tabled::settings::object::{Cell, Columns, Object, Rows, Segment};
use tabled::{
    grid::config::ColoredConfig,
    grid::records::{ExactRecords, PeekableRecords, Records},
    settings::{style::Style, themes::BorderCorrection, Alignment, Format, TableOption},
    Table, Tabled,
};

#[derive(Debug)]
pub struct MergeDuplicatesVerticalFirst;

impl<R, D> TableOption<R, ColoredConfig, D> for MergeDuplicatesVerticalFirst
where
    R: Records + PeekableRecords + ExactRecords,
{
    #[allow(clippy::assigning_clones)]
    // NOTE: Temporarily disabled due to a issue with `assigning_clones` not respecting MSRV in clippy 1.78.0.
    //       See https://github.com/rust-lang/rust-clippy/issues/12502
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

            // we need to mitigate messing existing spans
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

#[derive(Tabled)]
struct Editor<'a> {
    Ranking: &'a str,
    Function: &'a str,
    #[tabled(rename = "Latency (mean)")]
    Latency: usize,
}

#[derive(Tabled)]
pub struct ChangesTable {
    pub FunctionIdVs: String,
    #[tabled(rename = "Latency (mean)")]
    pub LatencyMean: String,
    #[tabled(rename = "Latency Change (mean)")]
    pub LatencyMeanChange: String,
    pub Result: String,
}

pub fn print_changes_table(group_id: &str, rows: &[ChangesTable]) {
    let mut table = Table::new(rows);

    table.modify((0, 0), Format::content(|_| group_id.to_string()));

    table
        .with(Style::modern_rounded())
        .with(MergeDuplicatesVerticalFirst)
        .with(BorderCorrection::span())
        .with(Alignment::center())
        .with(Alignment::center_vertical());

    println!("{table}");
}

fn tt() {
    #[rustfmt::skip]
    let data = [
        Editor { Ranking: "1", Function: "fast", Latency: 21 },
        Editor { Ranking: "1", Function: "fast2", Latency: 22 },
        Editor { Ranking: "1", Function: "fast3", Latency: 22 },
        Editor { Ranking: "2", Function: "original", Latency: 23 },
        Editor { Ranking: "3", Function: "alternative", Latency: 24 },
    ];

    // data[0].Latency.re
    let mut table = Table::new(data);

    table.modify((0, 0), Format::content(|s| format!(": {} :", s)));

    table.with(
        Style::modern_rounded(), // .horizontals([(1, HorizontalLine::inherit(Style::modern_rounded()))])
                                 // .verticals([(1, VerticalLine::inherit(Style::modern_rounded()))])
                                 // .remove_horizontal()
                                 // .remove_horizontals()
                                 // .remove_vertical()
                                 // .verticals([(1, VerticalLine::inherit(Style::modern()))]),
    );
    // table.with(Merge::vertical())
    // table.with(Columns::one(0), Merge::vertical());
    // table.with(BorderCorrection::span());
    // let sett = Settings::empty().with(Merge::vertical());
    // table.modify(Columns::one(0), &settings);
    // table.modify(Rows::first(), sett);
    table
        .with(MergeDuplicatesVerticalFirst)
        .with(BorderCorrection::span())
        .with(Alignment::center())
        .with(Alignment::center_vertical());
    println!("{table}");
}
