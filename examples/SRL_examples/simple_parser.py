import os

def simple_parser(filename):
    """
    Reads input file in the text format:
    ReceiverName\tPolarity\tError\tAzimuth\tTakeOffAngle\n
    """
    # One file per event
    with open(filename) as f:
        lines = f.readlines()
    # Set up polarity data dictionary
    polarity_dict = {'Stations': {'Name': [], 'TakeOffAngle': [],
                                  'Azimuth': []},
                   'Measured': [], 'Error': []}
    # Loop over lines
    for l in lines:
      # Check there are 5 data types
        if len(l.split('\t')) == 5:
            # Get data from line
            name, polarity, error, azimuth, to_angle = l.rstrip().split('\t')
            # Append data to the polarity_dict (and convert to correct format)
            polarity_dict['Stations']['Name'].append(name)
            polarity_dict['Stations']['TakeOffAngle'].append(float(to_angle))
            polarity_dict['Stations']['Azimuth'].append(float(azimuth))
            polarity_dict['Measured'].append(float(polarity))
            polarity_dict['Error'].append(float(error))
    # Filename as UID
    UID = os.path.splitext(os.path.split(filename)[-1])[0]
    event_dict = {'UID': UID, 'PPolarity': polarity_dict}
    # Return list of event dictionary
    return [event_dict]