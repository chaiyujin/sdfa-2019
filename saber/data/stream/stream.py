import numpy as np


def index_of(ts, tslist):
    left, right = 0, len(tslist)
    m = (left+right)//2
    while left < right:
        m = (left+right)//2
        tm = tslist[m]
        tn = tslist[m+1] if m + 1 < len(tslist) else ts + 1
        if tm <= ts < tn:
            break
        elif tm > ts:
            right = m
        else:  # ts >= tn
            left = m + 1
    return m


def seek(ts, timestamps, sequence):
    """ Timestamps should be in order """
    assert len(timestamps) == len(sequence)
    left, right = 0, len(timestamps)
    m = (left+right)//2
    while left < right:
        m = (left+right)//2
        tm = timestamps[m]
        tn = timestamps[m+1] if m + 1 < len(timestamps) else ts + 1
        if tm <= ts < tn:
            break
        elif tm > ts:
            right = m
        else:  # ts >= tn
            left = m + 1
    # check ts is in range
    if ts < timestamps[m] or ts > timestamps[-1]:
        # print("seek {} is out of range ({} ~ {})".format(
        #       ts, timestamps[0], timestamps[-1]))
        return np.copy(sequence[m])
    # return
    if m + 1 >= len(timestamps):
        return np.copy(sequence[m])
    else:
        n = m + 1
        a = (timestamps[n] - ts) / (timestamps[n] - timestamps[m])
        return a * sequence[m] + (1-a) * sequence[n]


def seek_subseq(length, start_ts, delta_ts, tslist, sequence):
    ret = []
    cur_m = index_of(start_ts, tslist)
    cur_t = start_ts
    for i in range(length):
        if cur_t < tslist[0]:
            ret.append(sequence[0])
        elif cur_t >= tslist[-1] or cur_m + 1 >= len(tslist):
            ret.append(sequence[-1])
        else:
            # maybe need update m
            while cur_t >= tslist[cur_m+1]:
                cur_m += 1
            if cur_m + 1 < len(tslist) and tslist[cur_m] <= cur_t < tslist[cur_m+1]:
                a = (tslist[cur_m+1] - cur_t) / (tslist[cur_m+1] - tslist[cur_m])
                ret.append(a * sequence[cur_m] + (1-a) * sequence[cur_m+1])
            else:
                ret.append(sequence[-1])
        cur_t += delta_ts
    return np.asarray(ret)
