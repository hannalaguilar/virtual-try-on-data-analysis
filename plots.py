import warnings
from typing import Optional
import pandas as pd
from math import pi
import numpy as np
from bokeh.io import export_png
from bokeh.palettes import Category20, Category10, TolRainbow, Bokeh
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.transform import cumsum


def plot_cat_distribution(df: pd.DataFrame):

    category_name = df.columns[0]
    df['angle'] = df['proportion'] * 2 * pi

    num_items = len(df)
    palette = None
    if num_items == 2:
        palette = ['#1f77b4', '#ff7f0e']
    elif num_items >= 3:
        palette = Category10[num_items]
    else:
        warnings.warn('There is only one segment in the data.', UserWarning)

    df['color'] = palette

    p = figure(height=350, width=500, title=f"{category_name.title()} ", toolbar_location=None,
               tools="hover", tooltips=f"{category_name}@: @proportion{0}%", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field=category_name, source=df)

    p.annular_wedge(x=0, y=1, inner_radius=0, outer_radius=0.2,
                    start_angle=0, end_angle=2 * pi, color="white")

    p.axis.visible = False
    p.grid.grid_line_color = None
    p.legend.location = "right"
    p.legend.label_text_font_size = "10pt"

    return p



def plot_viton_category_name(data: list, filename: Optional[str], save: bool = False) -> None:

    category_data = {item['file_name']: item['category_name'] for item in data}
    category_data_df = pd.DataFrame(list(category_data.items()), columns=['file_name', 'category_name'])
    category_counts = category_data_df['category_name'].value_counts().reset_index()
    category_counts.columns = ['category_name', 'count']

    # Define a fixed color map for the categories
    fixed_color_map = {
      'TOPS': '#4477AA',
      'WHOLEBODIES': '#EE6677',
      'SKIRTS': '#228833',
      'PANTS': '#CCBB44',
      'OUTWEARS': '#66CCEE',
    }

    # Create a data dictionary with category names and their counts, and format the numbers with commas
    data = {
      'categories': category_counts['category_name'].tolist(),
      'real_counts': category_counts['count'].tolist(),

      # Format numbers with commas for display
      'formatted_counts': [f"{count:,}" for count in category_counts['count']],  # Add commas

      # Fictitious values for bar heights
      'fictitious_sizes': [100, 8, 4, 2, 1],  # Fictitious values for bar heights

      # Add a column for colors based on the fixed color map
      'colors': [fixed_color_map[cat] for cat in category_counts['category_name']]  # Use the fixed color map
    }

    # Adjust x_offset dynamically based on the length of real_counts (for better centering)
    x_offsets = [-25 if count > 9999 else -10.5 for count in data['real_counts']]

    # Create a ColumnDataSource for the plot
    data['x_offsets'] = x_offsets
    source = ColumnDataSource(data=data)

    # Create the figure with more top margin
    p = figure(x_range=data['categories'], height=400, title="VITON-HD distribution",
              toolbar_location=None, tools="", y_range=(0, 110),
              output_backend='canvas')  # Increased y_range for more space

    # Create the bars using fictitious sizes and reference the 'colors' column for fill_color
    p.vbar(x='categories', top='fictitious_sizes', width=0.9, source=source, line_color='white',
          fill_color='colors')  # Reference the 'colors' column

    # Add labels with the formatted counts (with commas) on top of each bar
    labels = LabelSet(x='categories', y='fictitious_sizes', text='formatted_counts', level='glyph',
                     x_offset='x_offsets', y_offset=5, source=source)  # Adjusted y_offset for above-the-bar placement

    p.add_layout(labels)

    # Visual configuration
    p.xgrid.grid_line_color = None
    p.yaxis.visible = False  # Hide y-axis since the heights are fictitious
    p.xaxis.axis_label = 'Categories'

    if save:
      export_png(p, filename=filename)

    # Show the plot
    show(p)


# if __name__ == '__main__':
#    main()