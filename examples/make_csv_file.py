# !/usr/bin/env
"""make_csv_file.py

Commented example script to make CSV file
"""


def run(test=False):
    # Get data:
    from example_data import csv_data
    data = csv_data()
    if not isinstance(data, list):
        data = [data]
    # make csv file
    print('\nMaking CSV example file:')
    fname = 'csv_example_file.csv'
    import random  # To show header order doesn't matter
    output = []
    headers = ['Name', 'Azimuth', 'TakeOffAngle', 'Measured', 'Error']
    for event in data:
        output.append(','.join(['UID='+event.get('UID', '1'), '', '', '', '']))
        for key in event.keys():
            if key != 'UID':
                # Show header order doesn't matter
                random.shuffle(headers)
                output.append(','.join([key, '', '', '', '']))
                output.append(','.join(headers))
                for i, station_name in enumerate(event[key]['Stations']['Name']):
                    line = []
                    for head in headers:
                        if head in ['Azimuth', 'TakeOffAngle']:
                            line.append(str(float(event[key]['Stations'][head][i, 0])))
                        elif head == 'Name':
                            line.append(str(event[key]['Stations'][head][i]))
                        else:
                            line.append(str(event[key][head][i]).lstrip('[').lstrip().rstrip(']').rstrip())
                    output.append(','.join(line))
        output.append(','.join(['', '', '', '', '']))
    with open(fname, 'w') as f:
        f.write('\n'.join(output))

    # Output text
    print('\nSaved CSV example to: '+fname+'\n\n')
    print('CSV Example File:\n----------------------------------')
    print('\n'.join(output))
    print('-------------------------------------------\n')
    print('Events are split by blank lines.')
    print('Header order does not matter since the header line shows where the information is.')
    print('The UID and data-type information are stored in the first column.')
    print('\nThis CSV file contains '+str(len(data))+' events:')
    for i, ev in enumerate(data):
        def ordinal(n):
            return "%d%s" % (n, "tsnrhtdd"[(n/10 % 10 != 1)*(n % 10 < 4)*n % 10::4])
        if 'UID' in ev:
            print('\tThe '+str(ordinal(i+1))+' has UID: '+ev['UID']+' and data types: '+', '.join([key for key in ev.keys() if key != 'UID']))
        else:
            print('\tThe '+str(ordinal(i+1))+' has no UID and data types: '+', '.join([key for key in ev.keys() if key != 'UID']))

    # Test event load
    print('\n\nChecking that the csv data parses and is the same as the original data')

    from MTfit import inversion
    loaded_data = inversion.parse_csv(fname)
    data_check = check_data(loaded_data, data)
    print('\n\tLoaded data same as original data: {}'.format(data_check))
    return data_check


# Function to compare data structures


def check_data(loaded_data, original_data):
    import numpy as np
    output = True
    if not isinstance(loaded_data, original_data.__class__):
        print('Different type: {}, {}'.format(type(loaded_data), type(original_data)))
        return False
    if isinstance(loaded_data, list):
        if len(loaded_data) != len(original_data):
            print('Different size: {}, {}'.format(len(loaded_data), len(original_data)))
            return False
        for i, l_i in enumerate(loaded_data):
            output = (output and check_data(l_i, original_data[i]))
            if not output:
                return False
    elif isinstance(loaded_data, dict):
        if sorted(loaded_data.keys()) != sorted(original_data.keys()):
            print('Different keys: {}, {}'.format(sorted(loaded_data.keys()), sorted(original_data.keys())))
            return False
        for key in loaded_data.keys():
            output = (output and check_data(loaded_data[key], original_data[key]))
            if not output:
                return False
    elif isinstance(loaded_data, np.ndarray):
        output = (output and (loaded_data == original_data).all())
        if not output:
            print('Different values: {}, {}'.format(loaded_data, original_data, loaded_data == original_data))
    else:
        output = (output and loaded_data == original_data)
        if not output:
            print('Different values: {}, {}'.format(loaded_data, original_data, loaded_data == original_data))
    return output


if __name__ == "__main__":
    run()
