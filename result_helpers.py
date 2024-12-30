from xml.etree import ElementTree as ET
import re
from ledger_helpers import ResultsRow, Results, Cell
import numpy as np


def get_ns(root) -> str:
    """Get the namespace of the xml file."""
    ns = re.match(r"{.*}", root.tag)
    return ns.group(0) if ns else ""


def is_name(s: str) -> bool:
    """Check if a string is a name."""
    if not s:
        return False
    return (sum(i.isnumeric() for i in s) <= 2 and len(s) >= 3) or s.isalpha()


class LineOfText:
    region_points = []

    # Based on a TextLine in the xml file.
    def __init__(self, line, ns, offset=0):
        self.ns = ns
        self.line = line
        if not line:
            return

        # Text
        self.plain_text = line.find(f".//{ns}PlainText").text
        if not self.plain_text:
            self.plain_text = ""

        # Baseline points
        self.baseline = line.find(f".//{ns}Baseline")
        self.points = [
            point.split(",") for point in self.baseline.attrib["points"].split(" ")
        ]
        self.points = [(int(x) + offset, int(y)) for x, y in self.points]
        self.points.sort()

        # Region used for IOU calculation.
        self.region = line.find(f".//{ns}Coords")
        self.region_points = [
            point.split(",") for point in self.region.attrib["points"].split(" ")
        ]
        self.region_points = [(int(x) + offset, int(y)) for x, y in self.region_points]
        self.region_points.sort()

        self.avg_x = int(sum(x for x, _ in self.points) / len(self.points))
        self.avg_y = int(sum(y for _, y in self.points) / len(self.points))
        self.min_x = min([x for x, _ in self.points])
        self.max_x = max([x for x, _ in self.points])
        self.min_y = min([y for _, y in self.points])
        self.max_y = max([y for _, y in self.points])

        self.reg_avg_x = int(
            sum(x for x, _ in self.region_points) / len(self.region_points)
        )
        self.reg_avg_y = int(
            sum(y for _, y in self.region_points) / len(self.region_points)
        )
        self.reg_min_x = min([x for x, _ in self.region_points])
        self.reg_max_x = max([x for x, _ in self.region_points])
        self.reg_min_y = min([y for _, y in self.region_points])
        self.reg_max_y = max([y for _, y in self.region_points])

    def to_cell(self, top_offset=0, left_offset=0):
        pts = [(x + left_offset, y + top_offset) for x, y in self.points]
        return Cell(self.plain_text, pts)

    @staticmethod
    def from_string(s: str):
        t = LineOfText(None, None, 0)
        t.plain_text = s
        return t


def loghi_2_csv(file):
    # This function reads a pageXML file from Loghi and converts it to a csv file.

    # Open the pageXML file
    tree = ET.parse(file)

    if len(list(tree.iter())) < 20:
        print("File is empty", file)
        return None

    root = tree.getroot()

    ns = get_ns(root)
    lines = []
    for line in root.findall(f".//{ns}TextLine"):
        line = LineOfText(line, ns)
        lines.append(line)

    # Group the data into two columns based on the x-value
    page_avg_x = [l.avg_x for l in lines]
    avg_x = sum(page_avg_x) / len(page_avg_x)
    avg_x = (np.median(page_avg_x) + avg_x) / 2

    rows = [
        [l]
        for l in lines
        if l.avg_x < avg_x
        and any(i.isalpha() for i in l.plain_text)
        and len(l.plain_text) > 2
    ]
    right_item = [l for l in lines if l.avg_x >= avg_x]

    # Associate each item in the right_item set with a row in the left_item set
    for item in right_item:
        min_dist = 10**10  # arbitrary large number
        min_row = None

        for row in rows:
            dist = abs(item.avg_y - row[0].avg_y)
            if dist < min_dist:
                min_dist = dist
                min_row = row

        if min_row:
            min_row.append(item)
    rows = [sorted(row, key=lambda x: x.avg_x) for row in rows]
    rows.sort(key=lambda x: x[0].avg_y)

    # Convert the rows to a csv format and save it
    new_file_name = file.replace(".xml", "-naive.csv")
    data = []
    for row in rows:
        rr = ResultsRow.from_row([Cell(None)] + [l.to_cell() for l in row])
        data.append(rr)

    return Results(data, new_file_name)


def parse_ground_truth(xml_file, type="content"):
    # Parse the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the namespace
    ns = get_ns(root)

    # Assert that there are TableCell elements in the xml file
    if not root.findall(f".//{ns}TableCell"):
        print("No TableCell elements found in the xml file", xml_file)
        return

    # Loop over the tableregions
    tr = root.findall(f".//{ns}TableRegion")

    # Sort the table regions by the x coordinate
    tr.sort(
        key=lambda x: int(
            x.find(f".//{ns}Coords").attrib["points"].split()[0].split(",")[0]
        )
    )

    for i, region in enumerate(tr):
        if i > 2:
            break

        max_row = 0
        max_col = 0
        for cell in region.findall(f".//{ns}TableCell"):
            row = int(cell.attrib["row"])
            col = int(cell.attrib["col"])
            max_row = max(max_row, row)
            max_col = max(max_col, col)

        # Create an empty table
        table = [[Cell(None) for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        for cell in region.findall(f".//{ns}TableCell"):

            row = int(cell.attrib["row"])
            col = int(cell.attrib["col"])

            if cell.find(f".//{ns}TextEquiv"):
                text = cell.find(f".//{ns}TextEquiv").find(f".//{ns}Unicode").text
                points = (
                    cell.find(f".//{ns}TextLine")
                    .find(f".//{ns}Coords")
                    .attrib["points"]
                    .split(" ")
                )
                points = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in points]
                assert len(points) >= 4
                c = Cell(text, points)
                table[row][col] = c
            else:
                # Make a cell with just the coordinates
                points = cell.find(f".//{ns}Coords").attrib["points"].split(" ")
                points = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in points]
                assert len(points) == 4
                c = Cell(None, points)
                table[row][col] = c

        # # Check if the first colum is the only row with letters
        first_col = [row[-1].data for row in table if row[-1].data is not None]
        second_col = [row[-2].data for row in table if row[-2].data is not None]
        all_chars = "".join(first_col) + "".join(second_col)
        if any([c.isalpha() for c in all_chars]):
            table = [row[::-1] for row in table]

        r = []
        for row in table:
            rr = ResultsRow.from_row(row)
            r.append(rr)

        txt_i = "l" if i == 0 else "r"
        yield Results(r, xml_file.replace(".xml", f"-{txt_i}.csv"))
