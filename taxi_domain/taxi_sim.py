"""
Created by Rohan Paleja on January 8, 2020
Purpose:
"""
import numpy as np
from utils.global_utils import save_pickle
import os
# first data seed unknown, second tr:50, test: 100, third 100
np.random.seed(100)

def tree_policy(feature, mturkcode):
    at_airport = feature[0]
    at_village = feature[1]
    at_city = feature[2]
    wait_bit = feature[3]
    traffic_bit = feature[4]

    if mturkcode == 0:
        if at_airport:
            if traffic_bit == 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit == 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 3052998:
        if at_airport:
            if traffic_bit <= 2:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'
            else:
                return 'wait'
        else:
            if at_city:
                if traffic_bit <= 0:
                    return 'wait'
                else:
                    return 'wait'
            else:
                if at_village:
                    return 'go_to_city'
                else:
                    return 'wait'

    elif mturkcode == 3368824:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if at_airport:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            return 'wait'
    elif mturkcode == 3389240:
        if at_village:
            if wait_bit <= 1:
                return 'wait'
            else:
                if traffic_bit <= 1:
                    return 'go_to_village'
                else:
                    return 'wait'
        else:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 3830847:
        if traffic_bit <= 1:
            if at_city:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 5855749:
        if at_airport:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 7949399:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if at_city:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                return 'go_to_village'

    elif mturkcode == 9162576:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'
    elif mturkcode == 9737116:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'
            else:
                return 'wait'

    elif mturkcode == 11079945:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 11304188:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'
    elif mturkcode == 12162873:
        if at_village:
            if wait_bit <= 1:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if at_airport:
                return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 12247858:
        if at_village:
            if wait_bit <= 1:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if at_city:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                return 'go_to_village'

    elif mturkcode == 13463842:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 14580281:
        if at_airport:
            if traffic_bit <= 2:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'go_to_city'

    elif mturkcode == 16723702:
        if at_airport:
            if wait_bit <= 0:
                return 'wait'
            else:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_city:
                return 'wait'
            else:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'wait'

    elif mturkcode == 16867476:
        if at_airport:
            if traffic_bit <= 1:
                return 'wait'
            else:
                if traffic_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 17097053:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            return 'go_to_village'

    elif mturkcode == 18268603:
        if at_airport:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
        else:
            if at_city:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 18537392:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_city:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 20531020:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'wait'

        else:
            if traffic_bit <= 0:
                return 'go_to_village'
            else:
                return 'wait'

    elif mturkcode == 21023571:
        if at_airport:
            if traffic_bit <= 2:
                return 'wait'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 21076096:
        if at_airport:
            return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 21085311:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return "go_to_city"
            else:
                return 'wait'

    elif mturkcode == 23490621:
        if at_airport:
            return 'wait'
        else:
            if at_city:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                return 'go_to_city'

    elif mturkcode == 24294746:
        if wait_bit <= 0:
            return 'wait'
        else:
            return 'go_to_city'

    elif mturkcode == 25649961:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_city:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 25736852:
        if traffic_bit <= 0:
            return 'go_to_city'
        else:
            return 'go_to_village'

    elif mturkcode == 26261745:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 27054920:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 27683789:
        if at_airport:
            if traffic_bit <= 1:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            else:
                if traffic_bit <= 2:
                    return 'go_to_village'
                else:
                    return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 28326567:
        if at_village:
            if wait_bit <= 1:
                return 'wait'
            else:
                return 'go_to_city'

        else:
            if at_city:
                if traffic_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_airport:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 28747470:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            return 'wait'

    elif mturkcode == 31381055:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'


    elif mturkcode == 33357343:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 33910937:
        if traffic_bit <= 2:
            return 'go_to_city'
        else:
            return 'go_to_village'

    elif mturkcode == 34119214:
        if at_city:
            return 'wait'
        else:
            return 'go_to_city'

    elif mturkcode == 36933017:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 37256609:
        if at_city:
            return 'wait'
        else:
            if at_village:
                return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 37452489:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'go_to_city'

    elif mturkcode == 37622892:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 38005658:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                if traffic_bit <= 2:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 40454995:
        if at_airport:
            return 'wait'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'


    elif mturkcode == 41067349:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 41578990:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'

    elif mturkcode == 41679032:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if wait_bit <= 0:
                return 'wait'
            else:
                if at_city:
                    return 'go_to_village'
                else:
                    return 'wait'


    elif mturkcode == 41863608:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                if wait_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if at_city:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_airport:
                    return 'go_to_village'
                else:
                    return 'wait'

    elif mturkcode == 42552688:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            return 'wait'

    elif mturkcode == 44806666:
        if traffic_bit <= 2:
            if at_city:
                if traffic_bit <= 0:
                    return 'wait'
                else:
                    return 'wait'
            else:
                if traffic_bit <= 2:
                    return 'go_to_village'
                else:
                    return 'wait'
        else:
            if at_village:
                return 'wait'
            else:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 45289446:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 45779789:
        if at_airport:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 46864119:
        if traffic_bit <= 0:
            return 'go_to_city'
        else:
            return 'go_to_village'

    elif mturkcode == 49241263:
        if at_airport:
            if traffic_bit <= 1:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            else:
                if traffic_bit <= 2:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_city:
                if traffic_bit <= 2:
                    return 'go_to_village'
                else:
                    return 'wait'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 52669810:
        if at_airport:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_city:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 54687182:
        if traffic_bit <= 1:
            return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 55396098:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if at_city:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                return 'go_to_village'

    elif mturkcode == 55538376:
        if at_airport:
            return 'go_to_village'
        else:
            if at_city:
                return 'wait'
            else:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 56348626:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 58424372:
        if traffic_bit <= 1:
            if at_airport:
                return 'go_to_city'
            else:
                if at_village:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if at_airport:
                return 'go_to_village'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'wait'

    elif mturkcode == 58669417:
        if at_city:
            if traffic_bit <= 0:
                return 'wait'
            else:
                return 'go_to_village'
        else:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 58971768:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_city:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'

            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 60884760:
        if traffic_bit <= 0:
            return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 62210554:
        if traffic_bit <= 2:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                return 'wait'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 63245492:
        if at_airport:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 64013590:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 64047718:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 64273899:
        if traffic_bit <= 0:
            if at_city:
                return 'wait'
            else:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 64948032:
        if at_airport:
            if at_city:
                if traffic_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 0:
                    return 'go_to_village'
                else:
                    return 'go_to_city'

    elif mturkcode == 67175174:
        if traffic_bit <= 1:
            if at_city:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'go_to_village'
                else:
                    return 'go_to_city'
            else:
                return 'go_to_city'

    elif mturkcode == 68767705:
        if traffic_bit <= 0:
            if at_village:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'wait'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if at_airport:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'wait'
            else:
                if traffic_bit <= 2:
                    return 'go_to_city'
                else:
                    return 'wait'

    elif mturkcode == 70479205:
        if at_airport:
            if traffic_bit <= 1:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            else:
                if traffic_bit <= 2:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
        else:
            if at_city:
                if traffic_bit <= 0:
                    return 'wait'
                else:
                    return 'wait'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 70910541:
        if traffic_bit <= 1:
            if at_airport:
                return 'go_to_city'
            else:
                if at_village:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if at_airport:
                return 'go_to_village'
            else:
                return 'wait'

    elif mturkcode == 71820868:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 73510532:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'wait'

    elif mturkcode == 75573996:
        if at_city:
            return 'wait'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_airport:
                    return 'go_to_city'
                else:
                    return 'wait'

    elif mturkcode == 78092967:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 78140435:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'


    elif mturkcode == 81061178:
        if at_village:
            if wait_bit <= 1:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            return 'go_to_village'

    elif mturkcode == 81853845:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'wait'
        else:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_city'

    elif mturkcode == 82198579:
        if at_airport:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                return 'wait'
            else:
                if at_city:
                    return 'go_to_village'
                else:
                    'wait_for_passenger'

    elif mturkcode == 85472574:
        if at_city:
            if traffic_bit <= 2:
                return 'wait'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 2:
                    return 'go_to_city'
                else:
                    return 'go_to_village'

    elif mturkcode == 86199075:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if traffic_bit <= 0:
                return 'go_to_village'
            else:
                return 'go_to_city'

    elif mturkcode == 87401536:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                if wait_bit <= 0:
                    return 'go_to_village'
                else:
                    return 'wait'
        else:
            if wait_bit <= 1:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 88971413:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if traffic_bit <= 2:
                return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 89464883:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 89738911:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            return 'wait'

    elif mturkcode == 90531002:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 92593805:
        if at_village:
            if wait_bit <= 2:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if at_airport:
                return 'go_to_village'
            else:
                return 'wait'

    elif mturkcode == 93246403:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                return 'go_to_city'
        else:
            if at_city:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_airport:
                    return 'go_to_village'
                else:
                    return 'go_to_city'

    elif mturkcode == 93355064:
        if at_airport:
            if traffic_bit <= 1:
                if traffic_bit <= 0:
                    return 'go_to_city'
                else:
                    return 'wait'
            else:
                if traffic_bit <= 2:
                    return 'wait'
                else:
                    return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 0:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'

    elif mturkcode == 93837032:
        if at_airport:
            return 'wait'
        else:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'

    elif mturkcode == 95488721:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_city:
                return 'wait'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 95565368:
        if at_airport:
            if traffic_bit <= 2:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_city'
        else:
            if at_city:
                return 'wait'
            else:
                return 'go_to_city'

    elif mturkcode == 96462748:
        if at_airport:
            return 'go_to_city'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                return 'wait'

    elif mturkcode == 97097207:
        if at_airport:
            if traffic_bit <= 0:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                return 'wait'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'

    elif mturkcode == 97629586:
        if at_village:
            if wait_bit <= 0:
                return 'wait'
            else:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
        else:
            if at_airport:
                return 'go_to_city'
            else:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 98193743:
        if wait_bit <= 1:
            if traffic_bit <= 1:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_city'
        else:
            if wait_bit <= 2:
                if at_city:
                    return 'wait'
                else:
                    return 'go_to_village'
            else:
                if at_village:
                    return 'wait'
                else:
                    return 'go_to_city'
    elif mturkcode == 98747751:
        if traffic_bit <= 1:
            if traffic_bit <= 0:
                if at_village:
                    return 'go_to_city'
                else:
                    return 'go_to_village'
            else:
                if at_city:
                    return 'go_to_village'
                else:
                    return 'go_to_city'
        else:
            if traffic_bit <= 2:
                if traffic_bit <= 0:
                    return 'go_to_village'
                else:
                    return 'go_to_city'
            else:
                if at_airport:
                    return 'wait'
                else:
                    return 'go_to_city'

    elif mturkcode == 99072395:
        if at_airport:
            if traffic_bit <= 1:
                return 'go_to_city'
            else:
                return 'go_to_village'
        else:
            if at_village:
                if wait_bit <= 1:
                    return 'wait'
                else:
                    return 'go_to_city'
            else:
                if traffic_bit <= 1:
                    return 'go_to_city'
                else:
                    return 'go_to_village'

    return -1


class Location:
    def __init__(self, wait_time=None, traffic_time=None, can_see_from_outside=False, name='city'):
        self.name = name
        self.wait_time = wait_time
        self.traffic_time = traffic_time
        self.can_see_from_outside = can_see_from_outside


class Person:
    def __init__(self, location):
        self.location = location

    def set_location(self, loc):
        self.location = loc

    def get_wait_time(self):
        if self.location.can_see_from_outside == True:
            return self.location.wait_time
        else:
            return -1

    def do_action(self, action, locations):
        if action == 'wait':
            pass
        elif action == 'go_to_city':
            self.location = locations[0]
        elif action == 'go_to_village':
            self.location = locations[1]


def generate_feature_vector(person):
    feature = [0, 0, 0, 0, 0]
    location = person.location.name
    if location == 'airport':
        feature[0] = 1

    if location == 'village':
        feature[1] = 1

    if location == 'city':
        feature[2] = 1

    feature[3] = person.get_wait_time()

    feature[4] = person.location.traffic_time
    return feature


def main():
    action_dict = {"wait": 0, 'go_to_city': 1, 'go_to_village': 2}
    mturkcodes = [24294746,
                  44806666,
                  16723702,
                  7949399,
                  25649961,
                  87401536,
                  52669810,
                  58971768,
                  33357343,
                  93246403,
                  96462748,
                  70479205,
                  37622892,
                  95565368,
                  34119214,
                  41863608,
                  88971413,
                  64948032,
                  90531002,
                  64013590,
                  14580281,
                  56348626,
                  28747470,
                  78092967,
                  20531020,
                  18268603,
                  92593805,
                  41067349,
                  13463842,
                  55396098,
                  60884760,
                  27683789,
                  12247858,
                  89738911,
                  64273899,
                  68767705,
                  98747751,
                  89464883,
                  97629586,
                  16867476,
                  55538376,
                  67175174,
                  98193743,
                  54687182,
                  41679032,
                  3830847,
                  78140435,
                  21076096,
                  45289446,
                  58424372,
                  62210554,
                  38005658,
                  3389240,
                  18537392,
                  9737116,
                  27054920,
                  63245492,
                  95488721,
                  97097207,
                  71820868,
                  5855749,
                  3052998,
                  75573996,
                  21023571,
                  37452489,
                  85472574,
                  64047718,
                  3368824,
                  31381055,
                  21085311,
                  73510532,
                  11079945,
                  40454995,
                  45779789,
                  12162873,
                  26261745,
                  99072395,
                  81853845,
                  9162576,
                  28326567,
                  11304188,
                  23490621,
                  70910541,
                  49241263,
                  86199075,
                  82198579,
                  81061178,
                  58669417,
                  17097053,
                  33910937,
                  93355064,
                  93837032,
                  41578990,
                  42552688,
                  25736852,
                  36933017,
                  46864119,
                  37256609]

    states = [[] for _ in range(len(mturkcodes))]
    actions = [[] for _ in range(len(mturkcodes))]
    failed_list = []
    for e, each_code in enumerate(mturkcodes):
        for i in range(25):
            wait_time = np.random.randint(4)
            traffic_time = np.random.randint(4)
            city = Location(wait_time, traffic_time, False, 'city')
            village = Location(wait_time, traffic_time, True, 'village')
            airport = Location(wait_time, traffic_time, False, 'airport')
            locations = [city, village, airport]
            person = Person(np.random.choice([village, airport]))
            print('Starting at ', person.location.name)
            done = False
            num_actions = 0
            while not done:
                feature = generate_feature_vector(person)
                states[e].append(feature)
                action = tree_policy(feature, each_code)
                if action == -1:
                    print('here')
                print(action)
                actions[e].append(action_dict[action])
                person.do_action(action, locations)
                num_actions += 1
                if num_actions >= 5:
                    done = True
                    print(each_code, 'failed')
                    failed_list.append((e, each_code, i))


                if person.location == city:
                    done = True
                    print('DONE')
                    print('=' * 10)
                elif person.location == village and action == 'wait':
                    done = True
                    print('DONE')
                    print('=' * 10)
                else:
                    continue

    print('data generated')
    obj = [states, actions, failed_list, mturkcodes]
    save_pickle(os.curdir, obj, 'testing_data_from_all_users_2.pkl', want_to_print=True)


if __name__ == '__main__':
    main()
