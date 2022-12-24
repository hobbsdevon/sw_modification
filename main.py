import numpy as np

#outputs the traceback matrix and the scoring matrix
def score_traceback_matrixs(seq1, seq2, match, mismatch, gap):

    cols = len(seq1) + 1
    rows = len(seq2) + 1

    score_matrix = np.zeros((rows, cols))
    traceback_matrix = []

    for i in range(rows):                   #
        traceback_matrix.append([])         #Best I could do. Hope one of you can do better.
        for j in range(cols):               #This makes a nested list.
            traceback_matrix[i].append("")  #Numpy didn't like my use of strings matrixes.

    traceback_matrix[0][0] = "S"
    score_matrix[0][0] = 0

    for i in range(1, cols):
        score_matrix[0][i] = 0
        traceback_matrix[0][i] = "H"

    for i in range(1, rows):
        score_matrix[i][0] = 0
        traceback_matrix[i][0] = "V"

    for i in range(1, rows):

        for j in range(1, cols):

            vert = score_matrix[i-1][j] + gap
            hori = score_matrix[i][j-1] + gap

            diag = score_matrix[i-1][j-1] + (match if seq1[j - 1] == seq2[i - 1] else mismatch)
            max_val = max(0, vert, hori, diag)
            score_matrix[i][j] = max_val

            if vert == max_val:
                traceback_matrix[i][j] += "V"

            if hori == max_val:
                traceback_matrix[i][j] += "H"

            if diag == max_val:
                traceback_matrix[i][j] += "D"

            if traceback_matrix[i][j] == "":
                traceback_matrix[i][j] = "S"

    return score_matrix, traceback_matrix

def find_traceback(seq1, seq2, score_matrix, traceback_matrix, begin_index = [0,0]):

    max_index = np.unravel_index(np.argmax(score_matrix), np.shape(score_matrix))
    max_score = score_matrix[max_index[0]][max_index[1]]

    match_or_gap = find_traceback_recursive(score_matrix, traceback_matrix, max_index, ["",""], "")
    match_or_gap_list = match_or_gap.split(",")

    nested_match_or_gap_list = []
    end_indexes = []

    for i in match_or_gap_list:
        nested_match_or_gap_list.append(i.split(" "))

    for i in range(len(nested_match_or_gap_list)):

        seq1_count = 0
        seq2_count = 0
        aligns1 = list(nested_match_or_gap_list[i][0])
        aligns2 = list(nested_match_or_gap_list[i][1])

        for j in range(len(nested_match_or_gap_list[i][0])):

            if nested_match_or_gap_list[i][0][j] == "M":
                aligns1[j] = seq1[max_index[1]-seq1_count-1]
                seq1_count += 1

        for j in range(len(nested_match_or_gap_list[i][1])):

            if nested_match_or_gap_list[i][1][j] == "M":
                aligns2[j] = seq2[max_index[0]-seq2_count-1]
                seq2_count += 1

        nested_match_or_gap_list[i][0] = ''.join(aligns1)[::-1]
        nested_match_or_gap_list[i][1] = ''.join(aligns2)[::-1]
        # print("INFO")
        # print(max_index[0])
        # print(seq2_count)
        # print(max_index[1])
        # print(seq1_count)
        end_indexes.append([max_index[0]+begin_index[0], max_index[1]+begin_index[1]])
        if i == 0:
            start_index = [max_index[0] - seq2_count + begin_index[0],  max_index[1] - seq1_count + begin_index[1]]

    #start_index = [max_index[0] + begin_index[0], max_index[1] + begin_index[1]]

    return nested_match_or_gap_list, max_score, end_indexes, start_index


def find_traceback_recursive(score_matrix, traceback_matrix, position, main_trace, secondary_trace, select = 0):

    if score_matrix[position[0]][position[1]] == 0:
        return str(main_trace[0]) + " " + str(main_trace[1]) + secondary_trace

    if select == 0:
        if len(traceback_matrix[position[0]][position[1]]) > 1:
            for i in range(1, len(traceback_matrix[position[0]][position[1]])):
                secondary_trace += "," + find_traceback_recursive(score_matrix, traceback_matrix, position, main_trace.copy(), "", i)

    if traceback_matrix[position[0]][position[1]][select] == "D":
        main_trace[0] += "M"
        main_trace[1] += "M"
        return find_traceback_recursive(score_matrix, traceback_matrix, [position[0]-1, position[1]-1], main_trace, secondary_trace)

    if traceback_matrix[position[0]][position[1]][select] == "V":
        main_trace[0] += "_"
        main_trace[1] += "M"
        return find_traceback_recursive(score_matrix, traceback_matrix, [position[0]-1, position[1]], main_trace, secondary_trace)

    if traceback_matrix[position[0]][position[1]][select] == "H":
        main_trace[1] += "_"
        main_trace[0] += "M"
        return find_traceback_recursive(score_matrix, traceback_matrix, [position[0], position[1]-1], main_trace, secondary_trace)



def multi_homolog_smith_waterman_alg(seq1, seq2, match, mismatch, gap, iters):

    seq1_list = [seq1]
    seq1_index = [[0,len(seq1)-1]]
    seq2_list = [seq2]
    seq2_index = [[0,len(seq2)-1]]

    return_values = []

    for i in range(iters):

        score_matrix_list = []
        traceback_matrix_list = []
        max_value = 0

        for j, seq1_sec in enumerate(seq1_list):

            if len(score_matrix_list) <= j:
                score_matrix_list.append([])

            if len(traceback_matrix_list) <= j:
                traceback_matrix_list.append([])

            for k, seq2_sec in enumerate(seq2_list):

                if len(score_matrix_list[j]) <= k:
                    score_matrix_list[j].append([])

                if len(traceback_matrix_list[j]) <= k:
                    traceback_matrix_list[j].append([])

                score_matrix_list[j][k], traceback_matrix_list[j][k] = score_traceback_matrixs(seq1_sec, seq2_sec, match, mismatch, gap)

                if np.max(score_matrix_list[j][k]) > max_value:
                    max_value = np.max(score_matrix_list[j][k])
                    max_matrix = [j,k]
                    align_index = [seq2_index[k][0], seq1_index[j][0]]

        aligns, max_scores, end_indexes, start_index = find_traceback(seq1, seq2, score_matrix_list[max_matrix[0]][max_matrix[1]], traceback_matrix_list[max_matrix[0]][max_matrix[1]], align_index)
        print(end_indexes)
        print("ALIGNS")
        print(aligns)
        max_seq1_end = 0
        max_seq2_end = 0
        for k, value2 in enumerate(end_indexes):
            print("value2")
            print(value2)
            if value2[1] > max_seq1_end:
                print("HEY YEAH")
                max_seq1_end = value2[1]
                max_seq1_end_index = k
            if value2[0] > max_seq2_end:
                max_seq2_end = value2[0]
                max_seq2_end_index = k
        for j, value in enumerate(seq1_index):
            print("END INDEX")
            print(max_seq1_end_index)
            if seq1_index[j][1] > max_seq1_end_index:
                print("if seq1_index[j][1] > max_seq1_end_index:")
                print(seq1_index[j])
                seq1_index.insert(j,[seq1_index[j][0], start_index[1]])
                print(seq1_index[j])
                seq1_index[j+1][0] = max_seq1_end
                print([seq1_index[j][0], start_index[1]])
                seq1_list.insert(j, seq1[seq1_index[j][0]:seq1_index[j][1]])
                seq1_list[j + 1] = seq1[seq1_index[j+1][0]:seq1_index[j+1][1]]

                for i, seq in enumerate(seq1_list):
                    if seq == '':
                        print("DELETED")
                        seq1_index.pop(i)
                        seq1_list.pop(i)
                        i-=1
                print("Index")
                print(seq1_index)
                return_values.append([aligns.copy(), start_index.copy(), end_indexes.copy(), max_scores.copy(), seq1_index.copy(), seq1_list.copy()])
                break
    return return_values


#Test
print("\n")

seq1 = "ABZSFJGHVNAJKFHSJJSDFLIQOWVLKASDFPOIQWERJKLZLBCGHYHI"
seq2 = "IKLKIBAJKFSDFKABCASDFHJASDFJLKASDJFLFASDOIFUJHNKYTP"
score, traceback = score_traceback_matrixs(seq1, seq2, 2, -1, -1)
print(find_traceback(seq1, seq2, score, traceback), "\n")

seq1split = ["ABZSFJGHVNAJK", "FPOIQWERJKLZLBCGHYHI"]
seq2split = ["IKLKIBAJKFSDFKABCASD", "LFASDOIFUJHNKYTP"]
scoresplit0, tracebacksplit0 = score_traceback_matrixs(seq1split[0], seq2split[0], 2, -1, -1)
scoresplit1, tracebacksplit1 = score_traceback_matrixs(seq1split[0], seq2split[1], 2, -1, -1)
scoresplit2, tracebacksplit2 = score_traceback_matrixs(seq1split[1], seq2split[0], 2, -1, -1)
scoresplit3, tracebacksplit3 = score_traceback_matrixs(seq1split[1], seq2split[1], 2, -1, -1)
print(find_traceback(seq1split[0], seq2split[0], scoresplit0, tracebacksplit0))
print(find_traceback(seq1split[0], seq2split[1], scoresplit1, tracebacksplit1))
print(find_traceback(seq1split[1], seq2split[0], scoresplit2, tracebacksplit2))
print(find_traceback(seq1split[1], seq2split[1], scoresplit3, tracebacksplit3))

print("\n")
#print(multi_homolog_smith_waterman_alg(sequ1, sequ2, 2, -1, -1, 2))
