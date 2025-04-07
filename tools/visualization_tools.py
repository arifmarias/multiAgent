import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from typing import Dict, List, Union, Optional, Any
import re
import sys
import logging

def create_bar_chart(data: str, title: str, x_label: str, y_label: str, orientation: str, color: str, figsize: str) -> str:
    """
    Creates a bar chart from provided data and returns it as a base64-encoded PNG image.
    
    Args:
        data (str): JSON string containing data for the chart. Expected format:
                   For simple bar chart: [{"x": "Category1", "y": 10}, {"x": "Category2", "y": 20}, ...]
                   For grouped bar chart: [{"group": "Group1", "x": "Category1", "y": 10}, {"group": "Group1", "x": "Category2", "y": 20}, ...]
        title (str): Chart title
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        orientation (str): "vertical" for vertical bars or "horizontal" for horizontal bars
        color (str): Color for bars (ignored for grouped charts)
        figsize (str): Figure size as "width,height" in inches
        
    Returns:
        str: Base64-encoded PNG image of the bar chart
    """
    try:
        # Parse data from JSON
        data_list = json.loads(data)
        
        # Parse figsize
        fig_width, fig_height = map(float, figsize.split(','))
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Check if we have grouped data
        is_grouped = any('group' in item for item in data_list)
        
        if is_grouped:
            # Process grouped bar chart
            df = pd.DataFrame(data_list)
            groups = df['group'].unique()
            categories = df['x'].unique()
            
            x = np.arange(len(categories))
            width = 0.8 / len(groups)
            
            for i, group in enumerate(groups):
                group_data = df[df['group'] == group]
                offset = i - len(groups)/2 + 0.5
                
                if orientation == "vertical":
                    ax.bar(x + width * offset, group_data['y'], width, label=group)
                else:
                    ax.barh(x + width * offset, group_data['y'], width, label=group)
            
            if orientation == "vertical":
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
            else:
                ax.set_yticks(x)
                ax.set_yticklabels(categories)
                
            ax.legend()
            
        else:
            # Process simple bar chart
            x_values = [item['x'] for item in data_list]
            y_values = [item['y'] for item in data_list]
            
            if orientation == "vertical":
                ax.bar(x_values, y_values, color=color)
            else:
                ax.barh(x_values, y_values, color=color)
        
        # Set labels and title
        ax.set_title(title)
        
        if orientation == "vertical":
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        else:
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64-encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"

def create_line_chart(data: str, title: str, x_label: str, y_label: str, line_style: str, markers: str, color: str, figsize: str) -> str:
    """
    Creates a line chart from provided data and returns it as a base64-encoded PNG image.
    
    Args:
        data (str): JSON string containing data for the chart. Expected format:
                   For simple line chart: [{"x": 1, "y": 10}, {"x": 2, "y": 20}, ...]
                   For multi-line chart: [{"series": "Series1", "x": 1, "y": 10}, {"series": "Series1", "x": 2, "y": 20}, ...]
        title (str): Chart title
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        line_style (str): Style of the line: "solid", "dashed", "dotted", "dashdot"
        markers (str): Marker style, e.g., "o" for circles, "s" for squares, "^" for triangles, or "" for no markers
        color (str): Color for the line (ignored for multi-line charts)
        figsize (str): Figure size as "width,height" in inches
        
    Returns:
        str: Base64-encoded PNG image of the line chart
    """
    try:
        # Parse data from JSON
        data_list = json.loads(data)
        
        # Parse figsize
        fig_width, fig_height = map(float, figsize.split(','))
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Check if we have multi-series data
        is_multi_series = any('series' in item for item in data_list)
        
        if is_multi_series:
            # Process multi-line chart
            df = pd.DataFrame(data_list)
            series_names = df['series'].unique()
            
            for series in series_names:
                series_data = df[df['series'] == series]
                series_data = series_data.sort_values('x')
                ax.plot(series_data['x'], series_data['y'], label=series, marker=markers)
            
            ax.legend()
            
        else:
            # Process simple line chart
            df = pd.DataFrame(data_list)
            df = df.sort_values('x')
            ax.plot(df['x'], df['y'], linestyle=line_style, marker=markers, color=color)
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64-encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        return f"Error creating line chart: {str(e)}"

def create_pie_chart(data: str, title: str, figsize: str, explode: str, colors: str, show_percent: bool) -> str:
    """
    Creates a pie chart from provided data and returns it as a base64-encoded PNG image.
    
    Args:
        data (str): JSON string containing data for the chart. Expected format:
                   [{"label": "Category1", "value": 30}, {"label": "Category2", "value": 20}, ...]
        title (str): Chart title
        figsize (str): Figure size as "width,height" in inches
        explode (str): Comma-separated list of explode values for each slice (e.g., "0,0.1,0")
        colors (str): Comma-separated list of colors for slices (e.g., "red,blue,green")
        show_percent (bool): Whether to show percentage values on slices
        
    Returns:
        str: Base64-encoded PNG image of the pie chart
    """
    try:
        # Parse data from JSON
        data_list = json.loads(data)
        
        # Parse figsize
        fig_width, fig_height = map(float, figsize.split(','))
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Extract labels and values
        labels = [item['label'] for item in data_list]
        values = [item['value'] for item in data_list]
        
        # Parse explode values if provided
        explode_values = None
        if explode:
            explode_values = [float(x) for x in explode.split(',')]
            if len(explode_values) != len(labels):
                explode_values = None
        
        # Parse colors if provided
        color_values = None
        if colors:
            color_values = colors.split(',')
            if len(color_values) != len(labels):
                color_values = None
        
        # Create pie chart
        autopct = '%1.1f%%' if show_percent else None
        ax.pie(values, labels=labels, explode=explode_values, colors=color_values, 
               autopct=autopct, shadow=True, startangle=90)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Set title
        ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64-encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        return f"Error creating pie chart: {str(e)}"

def format_table_data(data: str, title: str, headers: str, highlight_max: bool, highlight_min: bool) -> str:
    """
    Formats data into a well-structured HTML table for presentation.
    
    Args:
        data (str): JSON string containing data for the table. Expected format:
                   [{"col1": "value1", "col2": "value2", ...}, {...}, ...]
        title (str): Table title
        headers (str): Comma-separated list of header names (optional, will use dict keys if not provided)
        highlight_max (bool): Whether to highlight maximum values in each numeric column
        highlight_min (bool): Whether to highlight minimum values in each numeric column
        
    Returns:
        str: HTML-formatted table
    """
    try:
        # Parse data from JSON
        data_list = json.loads(data)
        
        if not data_list:
            return "<p>No data available to format</p>"
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Use custom headers if provided
        if headers:
            header_list = headers.split(',')
            if len(header_list) == len(df.columns):
                df.columns = header_list
        
        # Apply highlighting for min/max values if requested
        def highlight_max_vals(s):
            is_numeric = pd.to_numeric(s, errors='coerce').notna()
            if highlight_max and is_numeric.all():
                return ['background-color: #a8d08d' if v == max(s) else '' for v in s]
            return ['' for _ in s]
        
        def highlight_min_vals(s):
            is_numeric = pd.to_numeric(s, errors='coerce').notna()
            if highlight_min and is_numeric.all():
                return ['background-color: #f8cbad' if v == min(s) else '' for v in s]
            return ['' for _ in s]
        
        # Apply styling
        if highlight_max or highlight_min:
            styles = []
            if highlight_max:
                styles.append(highlight_max_vals)
            if highlight_min:
                styles.append(highlight_min_vals)
            
            styled_df = df.style.apply(lambda x: pd.Series(
                [''] * len(x),
                index=x.index
            ))
            
            for style_func in styles:
                for col in df.columns:
                    styled_df = styled_df.apply(lambda x: style_func(x) if x.name == col else [''] * len(x))
            
            html = styled_df.to_html()
        else:
            html = df.to_html(index=False)
        
        # Add title if provided
        if title:
            html = f"<h3>{title}</h3>\n{html}"
        
        return html
    
    except Exception as e:
        return f"Error formatting table: {str(e)}"

def python_viz_executor(code: str) -> str:
    """
    Executes Python code for data visualization and returns the output as a base64-encoded PNG image.
    
    Args:
        code (str): Python code that uses matplotlib or other visualization libraries.
                   The code should create a visualization but not call plt.show().
        
    Returns:
        str: JSON string containing the visualization data or error information
    """
    try:
        # Create a namespace for execution
        namespace = {}
        
        # Capture stdout to get print statements
        console_output = io.StringIO()
        sys.stdout = console_output
        
        # Execute the code
        exec(code, namespace)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Capture the current figure
        if 'plt' in namespace:
            # Convert plot to base64-encoded image
            buffer = io.BytesIO()
            namespace['plt'].savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            namespace['plt'].close()
            
            # Return valid JSON with the image data
            return json.dumps({
                "image": image_base64,
                "format": "image/png",
                "console_output": console_output.getvalue()
            })
        else:
            # Return valid JSON with error information
            return json.dumps({
                "error": "No matplotlib plot was created in the code",
                "console_output": console_output.getvalue()
            })
            
    except Exception as e:
        # Reset stdout if exception occurs
        sys.stdout = sys.__stdout__
        
        # Return valid JSON with error information
        return json.dumps({
            "error": str(e),
            "console_output": console_output.getvalue() if 'console_output' in locals() else ""
        })

def create_fallback_data():
    """
    Creates fallback data when table extraction fails.
    
    Returns:
        str: JSON string with fallback data
    """
    fallback_data = [
        {"category": "Data 1", "value": 10},
        {"category": "Data 2", "value": 20},
        {"category": "Data 3", "value": 15}
    ]
    return json.dumps(fallback_data)
    
def extract_table_from_markdown(markdown_text: str) -> str:
    """
    Extracts table data from markdown text and converts it to a structured format.
    
    Args:
        markdown_text (str): Markdown text containing a table.
        
    Returns:
        str: JSON string of the extracted table data
    """
    try:
        # Handle empty input
        if not markdown_text or markdown_text.strip() == "":
            return json.dumps([{"category": "No Data", "value": 0}])
            
        # First try to extract escaped JSON with data_table
        json_pattern = r'"data_table":\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        json_match = re.search(json_pattern, markdown_text, re.DOTALL)
        
        if json_match:
            escaped_table = json_match.group(1)
            # Unescape the string
            table_text = escaped_table.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
            
            # Process the table rows
            rows = table_text.strip().split('\n')
            if len(rows) >= 2:  # Need at least header and one data row
                # Identify header row
                header_row_idx = 0
                for i, row in enumerate(rows):
                    if ('usage_date' in row.lower() or 'date' in row.lower()) and '|' in row:
                        header_row_idx = i
                        break
                
                # Extract headers
                header_row = rows[header_row_idx] if header_row_idx < len(rows) else rows[0]
                headers = []
                for part in header_row.split('|'):
                    part = part.strip()
                    if part and not part.isdigit():  # Skip index columns
                        headers.append(part)
                
                # Process data rows (skip header and any separator rows)
                data_rows = []
                for i in range(header_row_idx + 1, len(rows)):
                    row = rows[i]
                    if '|' not in row or '-|-' in row:
                        continue
                    
                    # Extract values, handling index column if present
                    parts = [p.strip() for p in row.split('|')]
                    
                    # Skip the first column if it's a numeric index
                    is_first_column_index = False
                    if len(parts) > 1 and parts[1].strip().isdigit():
                        is_first_column_index = True
                    
                    values = []
                    for j, part in enumerate(parts):
                        if part.strip():
                            # Skip the part if it's the index column
                            if is_first_column_index and j == 1:
                                continue
                            values.append(part.strip())
                    
                    if len(values) >= 2:  # Need at least two values
                        row_data = {}
                        for j, val in enumerate(values):
                            if j < len(headers):
                                key = headers[j]
                            else:
                                key = f"column_{j}"
                            
                            # Convert to appropriate type
                            try:
                                if val.replace('.', '', 1).replace('-', '', 1).isdigit():
                                    if '-' in val and len(val) == 10:  # Likely a date
                                        row_data[key] = val
                                    elif '.' in val:
                                        row_data[key] = float(val)
                                    else:
                                        row_data[key] = int(val)
                                else:
                                    row_data[key] = val
                            except (ValueError, TypeError):
                                row_data[key] = val
                        
                        if row_data:
                            data_rows.append(row_data)
                
                if data_rows:
                    logging.info(f"Successfully extracted {len(data_rows)} rows from JSON data_table")
                    return json.dumps(data_rows)
        
        # Next try to find a regular table pattern
        table_pattern = r'(\|(?:[^\|\n]*\|){2,}\n\|(?:[^\|\n]*\|){2,}(?:\n\|(?:[^\|\n]*\|){2,})+)'
        table_match = re.search(table_pattern, markdown_text, re.MULTILINE)
        
        if table_match:
            table_text = table_match.group(1)
            rows = table_text.strip().split('\n')
            
            # Need at least two rows
            if len(rows) >= 2:
                # Extract headers from the first row
                headers = []
                for part in rows[0].split('|'):
                    part = part.strip()
                    if part and not part.isdigit():  # Skip index columns
                        headers.append(part)
                
                # Process data rows (skip header)
                data_rows = []
                for i in range(1, len(rows)):
                    row = rows[i]
                    if '|' not in row or '-|-' in row:
                        continue
                    
                    # Extract values, handling index column if present
                    parts = [p.strip() for p in row.split('|')]
                    
                    # Determine if first column is an index
                    is_first_column_index = False
                    if len(parts) > 1 and parts[1].strip().isdigit():
                        is_first_column_index = True
                    
                    values = []
                    for j, part in enumerate(parts):
                        if part.strip():
                            # Skip if it's the index column
                            if is_first_column_index and j == 1:
                                continue
                            values.append(part.strip())
                    
                    if len(values) >= 2:  # Need at least two values
                        row_data = {}
                        for j, val in enumerate(values):
                            if j < len(headers):
                                key = headers[j]
                            else:
                                key = f"column_{j}"
                            
                            # Convert to appropriate type
                            try:
                                if val.replace('.', '', 1).replace('-', '', 1).isdigit():
                                    if '-' in val and len(val) == 10:  # Likely a date
                                        row_data[key] = val
                                    elif '.' in val:
                                        row_data[key] = float(val)
                                    else:
                                        row_data[key] = int(val)
                                else:
                                    row_data[key] = val
                            except (ValueError, TypeError):
                                row_data[key] = val
                        
                        if row_data:
                            data_rows.append(row_data)
                
                if data_rows:
                    logging.info(f"Successfully extracted {len(data_rows)} rows from direct table pattern")
                    return json.dumps(data_rows)
        
        # Try to extract date-value pairs
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+(?:\.\d+)?)'
        date_matches = re.findall(date_pattern, markdown_text)
        
        if date_matches:
            date_data = []
            for date, value in date_matches:
                try:
                    date_data.append({
                        "date": date,
                        "value": float(value)
                    })
                except ValueError:
                    pass
                    
            if date_data:
                logging.info(f"Extracted {len(date_data)} date-value pairs")
                return json.dumps(date_data)
        
        # Last resort - extract any numbers
        number_pattern = r'(\d+(?:\.\d+)?)'
        numbers = re.findall(number_pattern, markdown_text)
        
        if numbers and len(numbers) >= 2:
            data = []
            for i, num in enumerate(numbers[:10]):
                try:
                    data.append({
                        "category": f"Value {i+1}",
                        "value": float(num)
                    })
                except ValueError:
                    pass
            
            if data:
                logging.info("Created fallback data from numerical values")
                return json.dumps(data)
        
        # Ultimate fallback
        logging.warning("No data patterns found, returning fallback data")
        return json.dumps([{"category": "Sample", "value": 100}, {"category": "Data", "value": 50}])
        
    except Exception as e:
        logging.error(f"Error in extract_table_from_markdown: {str(e)}")
        return json.dumps([{"category": "Error", "value": 100}])

def process_data_for_visualization(data_str: str) -> str:
    """
    Process extracted data to ensure it's in a format suitable for visualization.
    
    Args:
        data_str (str): String representation of data (possibly JSON)
        
    Returns:
        str: Properly formatted JSON string ready for visualization
    """
    try:
        logging.info(f"Processing data for visualization, input length: {len(data_str)}")
        
        # First, check if input is already valid JSON
        try:
            json_data = json.loads(data_str)
            if isinstance(json_data, list) and len(json_data) > 0:
                logging.info("Input is already valid JSON list")
                return data_str
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Look for a table structure in Genie's output
        ascii_table_pattern = r'(\|[^\n]*\|[^\n]*\|[^\n]*(?:\n\|[^\n]*\|[^\n]*\|[^\n]*)+)'
        ascii_matches = re.findall(ascii_table_pattern, data_str, re.MULTILINE)
        
        for match in ascii_matches:
            # Process the matched text as a table
            lines = match.strip().split('\n')
            
            # Need at least two lines to make a valid table (header + data)
            if len(lines) < 2:
                continue
                
            # Extract headers from the first line
            headers = []
            header_parts = [part.strip() for part in lines[0].split('|') if part.strip()]
            for part in header_parts:
                headers.append(part)
                
            if not headers:
                continue
                
            # Process data rows
            data_rows = []
            for line in lines[1:]:  # Skip header line
                if '|' not in line:
                    continue
                    
                # Split by pipe and clean values    
                values = [val.strip() for val in line.split('|') if val.strip()]
                
                # Skip rows without enough data
                if len(values) < 2:
                    continue
                    
                # Create a dictionary for this row
                row_data = {}
                for i, value in enumerate(values):
                    # Use available header or create generic one
                    if i < len(headers):
                        header = headers[i]
                    else:
                        header = f"column_{i}"
                        
                    # Convert numeric values to numbers
                    try:
                        if '.' in value and value.replace('.', '', 1).isdigit():
                            row_data[header] = float(value)
                        elif value.isdigit():
                            row_data[header] = int(value)
                        else:
                            row_data[header] = value
                    except ValueError:
                        row_data[header] = value
                        
                data_rows.append(row_data)
            
            if data_rows:
                logging.info(f"Successfully extracted {len(data_rows)} rows from ASCII table")
                return json.dumps(data_rows)
            
        # Look for date-value pairs if no table was found
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+(?:\.\d+)?)'
        date_matches = re.findall(date_pattern, data_str)
        
        if date_matches:
            date_data = []
            for date, value in date_matches:
                try:
                    date_data.append({
                        "date": date,
                        "value": float(value)
                    })
                except ValueError:
                    continue
                    
            if date_data:
                logging.info(f"Extracted {len(date_data)} date-value pairs")
                return json.dumps(date_data)
        
        # Extract any numerical data as fallback
        number_pattern = r'(\d+(?:\.\d+)?)'
        numbers = re.findall(number_pattern, data_str)
        
        if numbers and len(numbers) >= 2:
            data = []
            for i, num in enumerate(numbers[:10]):
                try:
                    data.append({
                        "category": f"Value {i+1}",
                        "value": float(num)
                    })
                except ValueError:
                    pass
            
            if data:
                logging.info("Created fallback data from numerical values")
                return json.dumps(data)
        
        # Ultimate fallback
        logging.info("Using default fallback data")
        return json.dumps([
            {"category": "Sample", "value": 100},
            {"category": "Data", "value": 50}
        ])
        
    except Exception as e:
        logging.error(f"Error in process_data_for_visualization: {str(e)}")
        return json.dumps([
            {"category": "Sample", "value": 100},
            {"category": "Data", "value": 50}
        ])

def extract_data_from_genie_format(text: str) -> str:
    """
    Extract data from the specific format shown in the MLflow trace images.
    
    Args:
        text (str): Text containing data in the format seen in the images
        
    Returns:
        str: JSON string of the extracted data
    """
    try:
        # First, try to find JSON array pattern
        if "[" in text and "]" in text:
            array_pattern = r'\[\s*({.*?}(?:\s*,\s*{.*?})*)\s*\]'
            array_matches = re.search(array_pattern, text, re.DOTALL)
            
            if array_matches:
                try:
                    json_str = f"[{array_matches.group(1)}]"
                    # Validate it's proper JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass
        
        # Pattern specifically for the format in your images
        pattern = r'({"\s*usage_date\s*":\s*\d+\s*,\s*"\s*avg_currency_conversion_rate\s*":\s*"[^"}]*"\s*})'
        matches = re.findall(pattern, text)
        
        if matches:
            # Combine matches into a JSON array
            data_str = f"[{', '.join(matches)}]"
            try:
                # Validate and clean
                data = json.loads(data_str)
                return json.dumps(data)
            except json.JSONDecodeError:
                # If it fails, try a more lenient approach
                data_str = data_str.replace("'", '"')
                try:
                    json.loads(data_str)
                    return data_str
                except json.JSONDecodeError:
                    pass
        
        # Try extracting data in "key": value format
        key_value_pattern = r'"([^"]+)"\s*:\s*([^,}\s]+|"[^"]+")'
        kv_matches = re.findall(key_value_pattern, text)
        
        if kv_matches:
            # Group by pairs to create objects
            structured_data = []
            current_obj = {}
            
            for key, value in kv_matches:
                # Clean up the value
                cleaned_value = value.strip('"')
                
                # Try to convert to numeric if possible
                try:
                    if '.' in cleaned_value:
                        cleaned_value = float(cleaned_value)
                    else:
                        cleaned_value = int(cleaned_value)
                except ValueError:
                    pass
                
                current_obj[key] = cleaned_value
                
                # Every two pairs, create a new object
                if len(current_obj) == 2:
                    structured_data.append(current_obj)
                    current_obj = {}
            
            # Add any remaining pairs
            if current_obj:
                structured_data.append(current_obj)
                
            if structured_data:
                return json.dumps(structured_data)
        
        # Fallback to dummy data
        return json.dumps([{"x": 1, "y": 100}, {"x": 2, "y": 200}])
        
    except Exception as e:
        # Ensure we always return valid JSON
        return json.dumps([{"x": 1, "y": 100}, {"x": 2, "y": 200}])

def process_data_from_table_text(table_text: str) -> str:
    """
    Process table text from Genie's data_table field into structured JSON.
    
    Args:
        table_text: The table text from Genie's data_table field
        
    Returns:
        str: JSON string of the processed data
    """
    try:
        # Clean the input - handle escape characters
        cleaned_text = table_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        
        # Split into lines
        lines = cleaned_text.strip().split('\n')
        if len(lines) < 2:
            return None
            
        # Find header line
        header_line = None
        for line in lines:
            if '|' in line and ('usage_date' in line or 'date' in line.lower()):
                header_line = line
                break
        
        if not header_line:
            # No specific date header, use first line with pipes
            for line in lines:
                if '|' in line and len(line.split('|')) > 2:
                    header_line = line
                    break
        
        if not header_line:
            return None
            
        # Extract headers
        headers = []
        for part in header_line.split('|'):
            part = part.strip()
            if part and not part.isdigit():  # Skip numeric headers (likely indices)
                headers.append(part)
        
        # Process data rows
        data = []
        for line in lines:
            # Skip header or separator lines
            if line == header_line or '-|-' in line or not '|' in line:
                continue
                
            # Split by pipe and clean values
            parts = [p.strip() for p in line.split('|')]
            
            # Determine if first column is an index
            is_first_column_index = False
            if len(parts) > 1 and parts[1].strip().isdigit():
                is_first_column_index = True
            
            values = []
            for j, part in enumerate(parts):
                if part.strip():
                    # Skip if it's the index column
                    if is_first_column_index and j == 1:
                        continue
                    values.append(part.strip())
            
            if len(values) >= 2:  # Need at least 2 values for a meaningful row
                row_data = {}
                
                # Map values to headers
                for i, value in enumerate(values):
                    if i < len(headers):
                        header = headers[i]
                    else:
                        header = f"column_{i}"
                        
                    # Convert numeric values
                    try:
                        if value.replace('.', '', 1).isdigit():
                            if '.' in value:
                                row_data[header] = float(value)
                            else:
                                row_data[header] = int(value)
                        else:
                            row_data[header] = value
                    except (ValueError, TypeError):
                        row_data[header] = value
                
                data.append(row_data)
        
        if data:
            return json.dumps(data)
            
        return None
    except Exception as e:
        logging.error(f"Error processing table text: {str(e)}")
        return None
      
def connect_genie_to_visualization(genie_output):
    """
    Extract and process data from Genie for the visualization agent.
    
    Args:
        genie_output: Output from the Genie agent
        
    Returns:
        dict: Input suitable for the visualization agent
    """
    # Initialize variables
    content = ""
    sql_query = ""
    data_table = ""
    structured_data = None
    
    logging.info("Starting connect_genie_to_visualization")
    
    # First extract content from the genie_output
    if isinstance(genie_output, dict):
        if 'content' in genie_output:
            content = genie_output['content']
            logging.info(f"Found content in genie_output, length: {len(content)}")
        
        # Look for tool messages that might contain the JSON response
        if 'messages' in genie_output:
            for msg in genie_output['messages']:
                if msg.get('role') == 'tool' and 'content' in msg:
                    try:
                        tool_content = json.loads(msg['content'])
                        if 'sql_query' in tool_content:
                            sql_query = tool_content['sql_query']
                            logging.info(f"Found SQL query in tool message: {sql_query[:50]}...")
                        
                        if 'data_table' in tool_content:
                            data_table = tool_content['data_table']
                            logging.info(f"Found data_table in tool message, length: {len(data_table)}")
                    except (json.JSONDecodeError, TypeError):
                        pass
                        
            # If we didn't find in tool messages, look for Genie's content
            if not data_table and not sql_query:
                for msg in reversed(genie_output['messages']):
                    if msg.get('role') == 'assistant' and 'content' in msg:
                        if msg.get('name') == 'Genie':
                            content = msg['content']
                            logging.info(f"Found Genie content in messages, length: {len(content)}")
                            break
                        elif not content:
                            content = msg['content']
    
    # Try to extract JSON data from content
    if not data_table or not sql_query:
        # Try to parse JSON from content
        try:
            # Look for a complete JSON object in the content
            json_match = re.search(r'({[\s\S]*?})', content)
            if json_match:
                potential_json = json_match.group(1)
                try:
                    json_data = json.loads(potential_json)
                    logging.info(f"Successfully parsed JSON from content: {json_data.keys()}")
                    
                    if 'sql_query' in json_data and not sql_query:
                        sql_query = json_data['sql_query']
                        logging.info(f"Extracted SQL query from JSON: {sql_query[:50]}...")
                        
                    if 'data_table' in json_data and not data_table:
                        data_table = json_data['data_table']
                        logging.info(f"Extracted data_table from JSON, length: {len(data_table)}")
                except json.JSONDecodeError:
                    # Not valid JSON, try other patterns
                    pass
        except Exception as e:
            logging.error(f"Error extracting JSON from content: {str(e)}")
    
    # Process data_table if found
    if data_table:
        logging.info("Processing data_table")
        try:
            # Clean up the data table (handle escaping)
            cleaned_table = data_table.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
            
            # Split into rows and extract data
            rows = cleaned_table.strip().split('\n')
            if len(rows) >= 2:  # Need at least header and data
                # Identify header row (sometimes there's an index column)
                header_row_idx = 0
                for i, row in enumerate(rows):
                    if 'usage_date' in row.lower() or 'date' in row.lower():
                        header_row_idx = i
                        break
                
                # Extract headers
                header_row = rows[header_row_idx]
                headers = []
                for part in header_row.split('|'):
                    part = part.strip()
                    if part and not part.isdigit():  # Skip index columns
                        headers.append(part)
                
                # Skip the separator row if present (contains "----")
                data_start_idx = header_row_idx + 1
                if data_start_idx < len(rows) and "----" in rows[data_start_idx]:
                    data_start_idx += 1
                
                # Process data rows
                data_rows = []
                for i in range(data_start_idx, len(rows)):
                    row = rows[i]
                    if '|' not in row:
                        continue
                    
                    # Extract values, handling index column if present
                    parts = [p.strip() for p in row.split('|')]
                    
                    # Determine if first column is an index
                    is_first_column_index = False
                    for j, part in enumerate(parts):
                        if j == 1 and part.strip().isdigit():  # First column after initial empty split
                            is_first_column_index = True
                            break
                    
                    values = []
                    for j, part in enumerate(parts):
                        if part.strip():
                            # Skip if it's the index column
                            if is_first_column_index and j == 1:
                                continue
                            values.append(part.strip())
                    
                    if len(values) >= 2:  # Need at least date and value
                        row_data = {}
                        for j, val in enumerate(values):
                            if j < len(headers):
                                key = headers[j]
                            else:
                                key = f"column_{j}"
                            
                            # Convert to appropriate type
                            try:
                                if val.replace('.', '', 1).replace('-', '', 1).isdigit():
                                    if '-' in val and len(val) == 10:  # Likely a date
                                        row_data[key] = val
                                    elif '.' in val:
                                        row_data[key] = float(val)
                                    else:
                                        row_data[key] = int(val)
                                else:
                                    row_data[key] = val
                            except (ValueError, TypeError):
                                row_data[key] = val
                        
                        if row_data:
                            data_rows.append(row_data)
                
                if data_rows:
                    logging.info(f"Successfully extracted {len(data_rows)} rows from data_table")
                    structured_data = json.dumps(data_rows)
        except Exception as e:
            logging.error(f"Error processing data_table: {str(e)}")
    
    # If we still haven't found structured data, try extracting from content
    if not structured_data:
        logging.info("Falling back to extract_table_from_markdown")
        structured_data = extract_table_from_markdown(content)
    
    # Create instruction for visualization
    if structured_data and structured_data != "[]" and structured_data != "null":
        instruction = f"""
        Please create a visualization based on this data: {structured_data}
        
        {f"SQL Query: {sql_query}" if sql_query else ""}
        
        If the data appears to contain dates or time periods, create a line chart.
        If the data is categorical, create a bar chart.
        
        Make your best effort to create a clear and informative visualization.
        """
    else:
        # Ultimate fallback
        fallback_data = json.dumps([
            {"category": "Sample", "value": 100},
            {"category": "Data", "value": 50}
        ])
        logging.warning("No structured data found, using fallback data")
        instruction = f"""
        I couldn't extract structured data from the previous response.
        Please create a simple visualization with this sample data: {fallback_data}
        """
    
    # Return properly formatted message array
    return {
        "messages": genie_output.get("messages", []) + [
            {"role": "user", "content": instruction.strip()}
        ]
    }

def extract_genie_table(text: str) -> str:
    """
    Extract data specifically from Genie's formatted tables.
    
    Args:
        text: The text containing the Genie table output
        
    Returns:
        str: JSON string of extracted data or None if no data found
    """
    try:
        # First look for the data_table marker which is specific to Genie output
        data_table_pattern = r'"data_table"\s*:\s*"(.+?)(?:"\s*,|\"\s*$)'
        data_table_match = re.search(data_table_pattern, text, re.DOTALL)
        
        if data_table_match:
            table_text = data_table_match.group(1)
            # Replace escaped characters
            table_text = table_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
            
            # Process the table rows
            rows = table_text.strip().split('\n')
            if len(rows) < 3:  # Need header, separator, and at least one row
                return None
                
            # Find header row
            header_row = None
            for row in rows:
                if 'usage_date' in row and 'avg_currency_conversion_rate' in row:
                    header_row = row
                    break
            
            if not header_row:
                # Try to find any row with multiple pipe characters
                for row in rows:
                    if row.count('|') >= 2:
                        header_row = row
                        break
            
            if not header_row:
                return None
                
            # Extract headers
            headers = []
            for part in header_row.split('|'):
                part = part.strip()
                if part:
                    headers.append(part)
            
            if not headers:
                return None
                
            # Process data rows
            data = []
            for row in rows:
                # Skip header row or lines that don't contain pipe separators
                if row == header_row or '|' not in row:
                    continue
                    
                # Get values
                values = []
                for part in row.split('|'):
                    part = part.strip()
                    if part:
                        values.append(part)
                
                if len(values) >= 2:  # Need at least date and value
                    row_data = {}
                    
                    # Map values to headers
                    for i, value in enumerate(values):
                        if i < len(headers):
                            header = headers[i]
                        else:
                            header = f"column_{i}"
                            
                        # Try to convert numerical values
                        try:
                            if value.replace('.', '', 1).isdigit():
                                if '.' in value:
                                    row_data[header] = float(value)
                                else:
                                    row_data[header] = int(value)
                            else:
                                row_data[header] = value
                        except (ValueError, TypeError):
                            row_data[header] = value
                    
                    data.append(row_data)
            
            if data:
                logging.info(f"Successfully extracted {len(data)} rows from Genie data_table")
                return json.dumps(data)
        
        # If data_table approach failed, look for a table format in the general text
        # This pattern looks for lines with multiple pipe characters that appear to be a table
        table_pattern = r'(?:(?:\|\s*[\w -]+\s*)+\|[\s\n]*){2,}'
        table_match = re.search(table_pattern, text, re.DOTALL)
        
        if table_match:
            table_text = table_match.group(0)
            rows = table_text.strip().split('\n')
            
            # Find header row - typically the first row with pipe separators
            header_row = None
            for row in rows:
                if '|' in row:
                    header_row = row
                    break
            
            if not header_row:
                return None
                
            # Extract headers
            headers = []
            for part in header_row.split('|'):
                part = part.strip()
                if part:
                    headers.append(part)
            
            if not headers:
                return None
                
            # Process data rows
            data = []
            for row in rows:
                # Skip header or empty rows
                if row == header_row or '|' not in row:
                    continue
                    
                # Extract values
                values = []
                for part in row.split('|'):
                    part = part.strip()
                    if part:
                        values.append(part)
                
                if len(values) >= 2:  # Need at least two values to be meaningful
                    row_data = {}
                    
                    # Map values to headers
                    for i, value in enumerate(values):
                        if i < len(headers):
                            header = headers[i]
                        else:
                            header = f"column_{i}"
                            
                        # Try to convert numerical values
                        try:
                            if value.replace('.', '', 1).isdigit():
                                if '.' in value:
                                    row_data[header] = float(value)
                                else:
                                    row_data[header] = int(value)
                            else:
                                row_data[header] = value
                        except (ValueError, TypeError):
                            row_data[header] = value
                    
                    data.append(row_data)
            
            if data:
                logging.info(f"Successfully extracted {len(data)} rows from general table format")
                return json.dumps(data)
        
        # If no structured table found, look for date-value pairs specifically
        date_value_pattern = r'(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+(?:\.\d+)?)'
        matches = re.findall(date_value_pattern, text)
        
        if matches:
            data = []
            for date, value in matches:
                try:
                    data.append({
                        "date": date,
                        "value": float(value)
                    })
                except ValueError:
                    pass
                    
            if data:
                logging.info(f"Extracted {len(data)} date-value pairs")
                return json.dumps(data)
        
        return None
        
    except Exception as e:
        logging.error(f"Error extracting Genie table: {str(e)}")
        return None
       
def extract_genie_content(genie_output: dict) -> str:
    """
    Extract the content from Genie's output, specifically targeting the table data.
    
    Args:
        genie_output (dict): The full output from the Genie agent
        
    Returns:
        str: The extracted content or an empty string if nothing found
    """
    try:
        # Try to get content directly
        if 'content' in genie_output and genie_output['content']:
            return genie_output['content']
        
        # Try to extract from messages
        if 'messages' in genie_output:
            messages = genie_output['messages']
            # Look for tool messages which might contain the data
            for msg in reversed(messages):  # Start from latest messages
                if msg.get('role') == 'tool' and 'content' in msg:
                    try:
                        # Content might be JSON string
                        content_data = json.loads(msg['content'])
                        if 'data_table' in content_data:
                            return content_data['data_table']
                        elif 'sql_query' in content_data and 'response' in content_data:
                            return content_data['response']
                        else:
                            return msg['content']
                    except:
                        return msg['content']
                
                # Regular assistant or tool messages
                elif 'content' in msg and msg['content']:
                    return msg['content']
        
        return ""
    except Exception as e:
        print(f"Error extracting Genie content: {str(e)}")
        return ""
    
def parse_genie_response(genie_output):
    """
    Parse the Genie response to extract the relevant data for visualization.
    
    Args:
        genie_output (dict): The output from the Genie agent
        
    Returns:
        str: JSON string representation of the data or a fallback dataset
    """
    try:
        # First, try to get the content field
        if 'content' in genie_output and genie_output['content']:
            extracted = extract_table_from_markdown(genie_output['content'])
            if extracted and extracted != "null":
                return extracted
                
        # Next, look through messages to find tool responses
        if 'messages' in genie_output:
            for msg in reversed(genie_output['messages']):  # Start from the most recent
                if msg.get('role') == 'tool' and 'content' in msg:
                    try:
                        # Try to parse as JSON
                        tool_content = json.loads(msg['content'])
                        if 'data_table' in tool_content:
                            return json.dumps(tool_content['data_table'])
                        return json.dumps(tool_content)
                    except json.JSONDecodeError:
                        # If not JSON, try to extract tables
                        extracted = extract_table_from_markdown(msg['content'])
                        if extracted and extracted != "null":
                            return extracted
                
                # Also check assistant messages for table data
                elif msg.get('role') == 'assistant' and 'content' in msg:
                    extracted = extract_table_from_markdown(msg['content'])
                    if extracted and extracted != "null":
                        return extracted
        
        # If we still have no data, try looking for any pre-formatted JSON string
        if 'messages' in genie_output:
            for msg in genie_output['messages']:
                if 'content' in msg and isinstance(msg['content'], str):
                    # Look for something that looks like escaped JSON
                    json_pattern = r'(\{.*\})'
                    matches = re.findall(json_pattern, msg['content'])
                    for match in matches:
                        try:
                            # Try to parse it as JSON
                            data = json.loads(match)
                            return json.dumps(data)
                        except json.JSONDecodeError:
                            continue
        
        # Fallback to a simple dataset if nothing works
        return json.dumps([
            {"category": "No Data Found", "value": 100}
        ])
    
    except Exception as e:
        # Return a valid JSON string even if there's an error
        return json.dumps([
            {"category": "Error", "value": 100},
            {"category": "Parsing", "value": 50}
        ])
    
def generate_visualization_code(data_description: str, chart_type: str, specific_requirements: str) -> str:
    """
    Generates Python code to create a visualization based on given parameters.
    
    Args:
        chart_type (str): Type of chart to generate (e.g., "bar", "line", "pie", "scatter")
        data_description (str): Description of the data to visualize
        specific_requirements (str): Any specific requirements for the visualization
        
    Returns:
        str: Python code that can be executed by python_viz_executor
    """
    # This is a placeholder function that would typically call an LLM in a real implementation
    # For simplicity, we'll return template code for each chart type
    
    if chart_type.lower() == "bar":
        code = """
import matplotlib.pyplot as plt
import numpy as np

# Sample data - replace with your actual data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [15, 30, 45, 22]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='skyblue')
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.grid(axis='y', alpha=0.3)
"""
    elif chart_type.lower() == "line":
        code = """
import matplotlib.pyplot as plt
import numpy as np

# Sample data - replace with your actual data
x = np.linspace(0, 10, 20)
y = np.sin(x) * x

# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', linewidth=2, markersize=6, color='forestgreen')
plt.title('Line Chart')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.grid(True, alpha=0.3)
"""
    elif chart_type.lower() == "pie":
        code = """
import matplotlib.pyplot as plt

# Sample data - replace with your actual data
labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [15, 30, 45, 10]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
explode = (0.1, 0, 0, 0)  # explode 1st slice

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title('Pie Chart')
"""
    elif chart_type.lower() == "scatter":
        code = """
import matplotlib.pyplot as plt
import numpy as np

# Sample data - replace with your actual data
np.random.seed(42)
n = 50
x = np.random.rand(n) * 10
y = 2 * x + np.random.randn(n) * 2

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=100, c=x, cmap='viridis', alpha=0.7)
plt.colorbar(label='X value')
plt.title('Scatter Plot')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.grid(alpha=0.3)
"""
    else:
        code = """
import matplotlib.pyplot as plt
import numpy as np

# Create a default plot since the chart type wasn't recognized
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5, f"Chart type '{chart_type}' not recognized", 
         horizontalalignment='center', verticalalignment='center', 
         transform=plt.gca().transAxes, fontsize=14)
plt.title('Visualization')
plt.axis('off')
"""
    
    # Add a comment about the data description and requirements
    code += f"""
# Note: This is template code based on:
# Chart type: {chart_type}
# Data description: {data_description}
# Specific requirements: {specific_requirements}
"""
    
    return code