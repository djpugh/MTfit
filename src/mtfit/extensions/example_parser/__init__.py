import os


def simple_parser(filename):
    # One file per event
    lines = open(filename).readlines()
    # Set up polarity data dictionary
    polarity_dict = {'Stations': {'Name': [], 'TakeOffAngle': [], 'Azimuth': []}, 'Measured': [], 'Error': []}
    # Loop over lines
    for l in lines:
        # Check there are 5 data types
        if len(l.split('\t')) == 5:
            # Get data from line
            name, polarity, error, azimuth, takeoff_angle = l.rstrip().split('\t')
            # Append data to the polarity_dict (and convert to correct format)
            polarity_dict['Stations']['Name'].append(name)
            polarity_dict['Stations']['TakeOffAngle'].append(float(takeoff_angle))
            polarity_dict['Stations']['Azimuth'].append(float(azimuth))
            polarity_dict['Measured'].append(float(polarity))
            polarity_dict['Error'].append(float(error))
    # filename as UID
    event_dict = {'UID': os.path.splitext(os.path.split(filename)[-1])[0], 'PPolarity': polarity_dict}
    # Return list of event dictionary
    return [event_dict]
