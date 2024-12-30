from collections import namedtuple
import cv2
import numpy as np
import skimage

# Global variables
VERT_LINE_TRESHOLD = 0.4
VERT_LINE_MIN_DIST = 70


# Named Tuple for vertical lines, Subtotal lines.
VertLine = namedtuple("VertLine", ["top", "bottom"])
SubtotalLine = namedtuple("SubtotalLine", ["left", "right", "top", "bottom"])


class Ledger:
    id = None
    path = None
    cropped_top = 0
    cropped_bottom = 0
    cropped_left = 0
    cropped_right = 0

    def __init__(self, path: str = None, ledger_id: str = None):
        if path:
            self.path = path
            self.original_im = cv2.imread(path)

        if ledger_id is not None:
            self.id = ledger_id

    @property
    def contrast_im(self):
        if not hasattr(self, "_contrast_im"):
            im = cv2.adaptiveThreshold(
                cv2.cvtColor(self.cropped_im, cv2.COLOR_BGR2GRAY),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                51,
                3,
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self._contrast_im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        return self._contrast_im

    @property
    def cropped_im(self):
        if not hasattr(self, "_cropped_im"):
            self.remove_borders()
        return self._cropped_im

    @classmethod
    def from_image(cls, im: np.ndarray, ledger_id: str = None):
        ld = cls(ledger_id=ledger_id)
        ld.original_im = im
        ld._cropped_im = im
        return ld

    def aspect_ratio(self):
        """Return the aspect ratio of the image."""
        return self.original_im.shape[1] / self.original_im.shape[0]

    def remove_borders(self):
        # Remove the top and bottom borders
        average = self.original_im.mean(axis=1).mean(axis=1)
        average_color = (
            self.original_im[500:-500].mean(axis=1).mean(axis=1).mean() * 0.8
        )
        first_row1 = np.argmax(average > average_color)
        last_row1 = np.argmax(average[::-1] > average_color) + 1

        # Remove the left and right borders
        average = self.original_im.mean(axis=0).mean(axis=1)
        average_color = (
            self.original_im.mean(axis=0)[500:-500].mean(axis=1).mean() * 0.8
        )
        first_row2 = np.argmax(average > average_color)
        last_row2 = np.argmax(average[::-1] > average_color) + 1

        # Save the cropped image
        self.cropped_top = first_row1
        self.cropped_bottom = last_row1
        self.cropped_left = first_row2
        self.cropped_right = last_row2

        self._cropped_im = self.original_im[
            first_row1:-last_row1, first_row2:-last_row2
        ]

    def is_double_page(self):
        return self.aspect_ratio() > 1

    def split_into_two(self):
        assert self.is_double_page(), "Ledger is not double page"

        half = self.cropped_im.shape[1] // 2
        left = self.cropped_im[:, :half]
        right = self.cropped_im[:, half:]
        return Ledger.from_image(left, ledger_id=f"{self.id}l"), Ledger.from_image(
            right, ledger_id=f"{self.id}r"
        )

    def horizontal_split(self, split_at: int, suf1="-a", suf2="-b"):
        # Split the ledger horizontally at the given row.
        # Return two ledgers, one above the row and one below the row.
        top = self.cropped_im[:split_at]
        bottom = self.cropped_im[split_at:]
        l1 = Ledger.from_image(top, ledger_id=f"{self.id}{suf1}")
        l2 = Ledger.from_image(bottom, ledger_id=f"{self.id}{suf2}")
        return l1, l2

    def veritcal_lines(self):
        if not self.vert_lines:
            self.find_vertical_lines()
        return self.vert_lines

    def find_vertical_lines(self):
        height, width = self.contrast_im.shape
        deciwidth = width // 60
        row_start = 0
        row_end = height - 2

        # Pre-compute top_pixel and delta combinations
        top_pixel_range = np.arange(deciwidth + 1, width - deciwidth, 5)
        delta_range = np.arange(-deciwidth, deciwidth, 5)
        top_pixels, deltas = np.meshgrid(top_pixel_range, delta_range, indexing="ij")

        # Compute bottom_pixel for all combinations
        bottom_pixels = top_pixels + deltas

        # Initialize results
        res = []

        # Iterate through precomputed pixel combinations
        for top_pixel, bottom_pixel in zip(top_pixels.ravel(), bottom_pixels.ravel()):
            rr, cc = skimage.draw.line(row_start, top_pixel, row_end, bottom_pixel)
            valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr, cc = rr[valid_indices], cc[valid_indices]

            # Fetch pixel values and count black pixels
            pxls = self.contrast_im[rr, cc]
            black_pixels = np.sum(pxls == 0)
            res.append((black_pixels, top_pixel, bottom_pixel))

        drawn = []
        first_line_count = 0

        for count, top_pixel, bottom_pixel in sorted(res, reverse=True):
            if first_line_count == 0:
                first_line_count = count
            if count / first_line_count < VERT_LINE_TRESHOLD:
                break

            if any(abs(top_pixel - p[0]) < VERT_LINE_MIN_DIST for p in drawn):
                continue

            if any(abs(bottom_pixel - p[1]) < VERT_LINE_MIN_DIST for p in drawn):
                continue

            # Cannot cross another line
            if any((top_pixel < p[0] and bottom_pixel > p[1]) for p in drawn):
                continue
            vl = VertLine(top_pixel, bottom_pixel)
            drawn.append(vl)
            if len(drawn) >= 12:
                break

        self._vert_lines = sorted(drawn, key=lambda x: x.top)

    @property
    def vert_lines(self):
        if not hasattr(self, "_vert_lines"):
            self.find_vertical_lines()
        return self._vert_lines

    @property
    def middle_line(self):
        if not hasattr(self, "_middle_line"):
            self._middle_line = self.vert_lines[self.middle_line_index]
        return self._middle_line

    @property
    def middle_line_index(self):
        if not hasattr(self, "_middle_line_index"):
            self.find_middle_line()
        return self._middle_line_index

    def find_middle_line(self):
        lines = [(i[0] + i[1]) / 2 for i in sorted(self.vert_lines)]

        # Find the 6 lines that are closest together. This is the middle line.
        zipped = np.array(list(zip(lines, lines[5:-3])))
        dif = [i[1] - i[0] for i in zipped]
        min_diff = np.argmin(dif)

        self._middle_line_index = min_diff + 4

    def check_vertical_lines(self):
        # This functions ensures that the vertical lines are correctly detected. Any extra lines are removed.

        # The first and second line should have the largest gap. Find the lines with the largest gap.
        gaps = [
            self.vert_lines[i + 1].top - self.vert_lines[i].bottom
            for i in range(len(self.vert_lines) - 1)
        ]
        max_gap = np.argmax(gaps)

        # Remove all lines before line 1
        self._vert_lines = self.vert_lines[max_gap:]

    def find_subtotal_lines(self, treshold=0.8):
        left_line = self.vert_lines[1]
        left_v = int((left_line.top + left_line.bottom) / 2)
        right_line = self.vert_lines[3]
        right_v = int((right_line.top + right_line.bottom) / 2)

        res = []
        offset = 50
        for row in range(offset, self.contrast_im.shape[0] - offset, 2):
            for right_row in range(row - offset, row + offset, 2):
                line = skimage.draw.line(right_row, left_v, row, right_v)
                black_pixels = np.count_nonzero(self.contrast_im[line] == 0)
                res.append((black_pixels, row, right_row))

        drawn_lines = []
        first_cnt = 0

        for cnt, row1, row2 in sorted(res, reverse=True, key=lambda x: x[0]):
            if first_cnt == 0:
                first_cnt = cnt

            if cnt / first_cnt < treshold:
                break

            if any(abs(row1 - p[2]) < 30 for p in drawn_lines):
                continue

            sl = SubtotalLine(left_line, right_line, row1, row2)
            drawn_lines.append(sl)

        self._subtotal_lines = sorted(drawn_lines, key=lambda x: x[0])

    @property
    def subtotal_lines(self):
        if not hasattr(self, "_subtotal_lines"):
            self.find_subtotal_lines()
        return self._subtotal_lines


class Region:
    r: list[tuple[int, int]] = None

    def __init__(self, r: list[tuple[int, int]]):
        self.r = r

    def __str__(self):
        return f"Region: {self.r}"

    def __repr__(self):
        return f"Region({self.r})"

    def iou(self, other):
        if self.r is None and other.r is None:
            return None
        if self.r is None or other.r is None:
            return 0

        r1 = np.array(self.r)
        r2 = np.array(other.r)

        # Get the bounding box of the two regions
        r1_x1 = np.min(r1[:, 0])
        r1_x2 = np.max(r1[:, 0])
        r1_y1 = np.min(r1[:, 1])
        r1_y2 = np.max(r1[:, 1])

        r2_x1 = np.min(r2[:, 0])
        r2_x2 = np.max(r2[:, 0])
        r2_y1 = np.min(r2[:, 1])
        r2_y2 = np.max(r2[:, 1])

        # Calculate the area of the two regions
        r1_area = (r1_x2 - r1_x1) * (r1_y2 - r1_y1)
        r2_area = (r2_x2 - r2_x1) * (r2_y2 - r2_y1)

        # Calculate the intersection
        intersection_x1 = max(r1_x1, r2_x1)
        intersection_x2 = min(r1_x2, r2_x2)
        intersection_y1 = max(r1_y1, r2_y1)
        intersection_y2 = min(r1_y2, r2_y2)

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(
            0, intersection_y2 - intersection_y1
        )

        # Calculate the union
        union_area = r1_area + r2_area - intersection_area

        return intersection_area / union_area


class Cell:
    data: any = ""
    region: Region = None

    def __init__(self, data, region=None):
        self.data = data
        if type(region) == list:
            self.region = Region(region)
        else:
            self.region = region

    def __str__(self):
        return f"Cell:({self.data}, {self.region})"

    def __repr__(self):
        return f"Cell({self.data})"


def calulcate_cer(a: str, b: str) -> float:
    """
    Calculate the Character Error Rate (CER) between two strings.

    CER = (Insertions + Deletions + Substitutions) / Total characters in the reference

    Args:
        a (str): The reference string.
        b (str): The hypothesis string (predicted).

    Returns:
        float: The CER value (0.0 means perfect match, higher means worse).
    """
    if not a and not b:
        return 0.0

    if not a or not b:
        return 1.0

    # Create a matrix for edit distance calculation (Levenshtein distance)
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Populate the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Deletion
                    dp[i][j - 1],  # Insertion
                    dp[i - 1][j - 1],
                )  # Substitution

    # Calculate CER
    levenshtein_distance = dp[m][n]
    cer = levenshtein_distance / m if m > 0 else float("inf")  # Avoid division by zero
    return min(1, cer)


class ResultsRow:
    """This class is used to store the results of one row of the ledger extraction.

    Can be used to compare to another ResultsRow, either for a word error rate or for IOU.
    """

    date: Cell = None
    account_name: Cell = None
    account_number: Cell = None
    amount1: Cell = None
    amount2: Cell = None
    amount3: Cell = None

    def __init__(self, date, account_name, account_number, amount1, amount2, amount3):
        self.date = date
        self.account_name = account_name
        self.account_number = account_number
        self.amount1 = amount1
        self.amount2 = amount2
        self.amount3 = amount3

    def is_subtotal(self):
        return (
            self.account_name.data is None
            and self.account_number.data is None
            and self.date.data is None
        )

    @staticmethod
    def from_row(row) -> "ResultsRow":
        row += [Cell(None)] * (6 - len(row))
        return ResultsRow(row[0], row[1], row[2], row[3], row[4], row[5])

    def __str__(self):
        return f"ResultsRow(date={self.date.data}, name={self.account_name.data}, an={self.account_number.data}, a1={self.amount1.data}, a2={self.amount2.data}, a3={self.amount3.data})"

    def __repr__(self):
        return f"rr({self.date.data},{self.account_name.data},{self.account_number.data},{self.amount1.data},{self.amount2.data},{self.amount3.data})"

    def compare_cer(self, other) -> float:
        # Compare the character error rate of two rows.
        an = calulcate_cer(self.account_number.data, other.account_number.data)
        a1 = calulcate_cer(self.amount1.data, other.amount1.data)
        date = calulcate_cer(self.date.data, other.date.data)

        if not an and not a1 and not date:
            return 1

        if not an:
            return (a1 + date) / 2
        if not a1:
            return (an + date) / 2
        if not date:
            return (an + a1) / 2

        return (an + a1 + date) / 3

    def compare_iou(self, other):
        if self.account_number == None or other.account_number == None:
            an = None
        elif self.account_number.data == None or other.account_number.data == None:
            an = None
        else:
            an = self.account_number.region.iou(other.account_number.region)

        if self.amount1 == None or other.amount1 == None:
            a1 = None
        elif self.amount1.data == None or other.amount1.data == None:
            a1 = None
        else:
            a1 = self.amount1.region.iou(other.amount1.region)

        if self.account_name == None or other.account_name == None:
            aname = None
        elif self.account_name.data == None or other.account_name.data == None:
            aname = None
        else:
            aname = self.account_name.region.iou(other.account_name.region)

        if an == None and a1 == None and aname == None:
            return None

        if an == None and a1 == None:
            return aname
        if an == None and aname == None:
            return a1
        if a1 == None and aname == None:
            return an

        if an == None:
            return (a1 + aname) / 2
        if a1 == None:
            return (an + aname) / 2
        if aname == None:
            return (an + a1) / 2

        return (an + a1 + aname) / 3


class Results:
    # This class is used to store the results of the ledger extraction.
    # Can be used to save to csv and compare with ground truth.
    data: list[ResultsRow] = None
    name: str = None

    def __init__(self, data: list[ResultsRow], name: str = None):
        self.data = data
        self.name = name

    def subtotals(self):
        st = [x.amount1 for x in self.data if x.is_subtotal()]
        st.sort(key=lambda x: x.region.r[0][1])
        return st

    def to_csv(self):
        """
        Converts the ledger data to a CSV formatted string.

        Iterates through each row in the ledger data and concatenates the
        values of date, account_name, account_number, amount1, amount2,
        and amount3, separated by commas. Each row is followed by a newline
        character.

        Returns:
            str: A string representing the ledger data in CSV format.
        """
        res = ""
        for row in self.data:
            res += row.date.data if row.date.data else ""
            res += ","
            res += row.account_name.data if row.account_name.data else ""
            res += ","
            res += row.account_number.data if row.account_number.data else ""
            res += ","
            res += row.amount1.data if row.amount1.data else ""
            res += ","
            res += row.amount2.data if row.amount2.data else ""
            res += ","
            res += row.amount3.data if row.amount3.data else ""
            res += "\n"
        return res

    def _compare(self, other, compare_func):
        options = []
        for o1, o2 in [(0, 1), (1, 0), (0, 0), (2, 0), (0, 2)]:
            res = []
            for i, j in zip(self.data[o1:], other.data[o2:]):
                r = compare_func(i, j)
                if r is not None:
                    res.append(r)
            if len(res) > 0:
                options.append(sum(res) / len(res))
        if not options:
            return None
        return options

    def compare_cer(self, other):
        o = self._compare(other, ResultsRow.compare_cer)
        if not o:
            return None
        return min(o)

    def compare_iou(self, other):
        o = self._compare(other, ResultsRow.compare_iou)
        if not o:
            return None
        return max(o)

    def compare_best_cer(self, other):
        # For each row, find the best fit row in the other ledger.
        res = []
        for row in self.data:
            best_fit_score = 0
            best_other_row = None
            for other_row in other.data:
                score = row.compare_iou(other_row)
                if score is None:
                    continue
                if score > best_fit_score:
                    best_fit_score = score
                    best_other_row = other_row
            if best_other_row:
                res.append(row.compare_cer(best_other_row))

        return sum(res) / len(res) if res else None

    def comapre_best_iou(self, other, th=None):
        # For each row, find the best fit row in the other ledger.
        res = []
        for row in self.data:
            best_fit_score = 0
            best_fit = None
            for other_row in other.data:
                score = row.compare_iou(other_row)
                if score is None:
                    continue
                if score > best_fit_score:
                    best_fit = other_row
                    best_fit_score = score
            if th:
                best_fit_score = 1 if best_fit_score > th else best_fit_score
            if best_fit:
                # print(best_fit, row)
                res.append(best_fit_score)

        return sum(res) / len(res) if res else None

    def __str__(self):
        return f"Results: {self.name} with {len(self.data)} rows"
