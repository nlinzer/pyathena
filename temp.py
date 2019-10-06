import re
import pandas as pd
import numpy as np

def _parse_line(rx, line):
    """
    Do a regex search against given regex and
    return the match result.

    """

    match = rx.search(line)
    if match:
        return match
    # if there are no matches
    return None

def parse_file(filepath):
    """
    Parse text at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file_object to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            rx = re.compile(r't=([0-9\.]+).*x=\((-?\d+),(-?\d+),(-?\d+)\).*n=([0-9\.]+).*nth=([0-9\.]+).*P=([0-9\.]+).*cs=([0-9\.]+)')
            match = _parse_line(rx, line)
            if match:
                time = float(match[1])
                x = float(match[2])
                y = float(match[3])
                z = float(match[4])
                rho = float(match[5])
                rho_crit = float(match[6])
                prs = float(match[7])
                cs = float(match[8])
                line = file_object.readline()
                rx = re.compile(r'navg=(-?[0-9\.]+).*id=(\d+).*m=(-?[0-9\.]+).*nGstars=(\d+)')
                match = _parse_line(rx, line)
                navg = float(match[1])
                idstar = int(match[2])
                mstar = float(match[3])
                nGstars = int(match[4])
                if (mstar > 0):
                    row = {
                        'time': time,
                        'x': x,
                        'y': y,
                        'z': z,
                        'rho': rho,
                        'rho_crit': rho_crit,
                        'prs': prs,
                        'cs': cs,
                        'navg': navg,
                        'idstar': idstar,
                        'mstar': mstar,
                        'nGstars': nGstars
                    }
                    data.append(row)
            else:
                line = file_object.readline()

        # create a pandas DataFrame from the list of dicts
        data = pd.DataFrame(data)
        # set the School, Grade, and Student number as the index
        data.sort_values('time', inplace=True)
    return data


#sf_data = parse_file("sf.txt")
