#!/usr/bin/env python3

def distance(s1, s2):
    dist1 = list(range(len(s1)+1))
    for j in range(len(s2)):
        # print(dist1)
        dist2 = [0]*(len(s1)+1)
        dist2[0] = dist1[0]+1
        for i in range(len(s1)):
            # print(f"{i},{dist1[i]}: {s2[j]},{s1[i]} t: {dist1[i+1]+1} l:{dist2[i]+1} d:{dist1[i]}")
            best = dist1[i+1]+1
            if dist2[i]+1 < best: best = dist2[i]+1
            if s1[i] == s2[j] and dist1[i] < best: best = dist1[i]
            dist2[i+1] = best
        dist1 = dist2
    # print(dist1)
    return dist1[-1]

def similarity(s1, s2):
    return 1.0-distance(s1, s2) / (len(s1) + len(s2))

if __name__ == '__main__':
    import sys
    text1 = sys.argv[1]
    text2 = sys.argv[2]
    print(f"d={distance(text1, text2)}")
    print(f"s={similarity(text1, text2)}")
