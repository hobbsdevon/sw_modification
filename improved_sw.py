import numpy as np

def score_traceback_matrixs(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Build the score and traceback matrices for Smith-Waterman algorithm.
    """
    m, n = len(seq1), len(seq2)
    score = np.zeros((m + 1, n + 1), dtype=int)
    traceback = np.zeros((m + 1, n + 1), dtype=int)  # 0: diagonal, 1: up (gap in seq2), 2: left (gap in seq1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag_score = score[i - 1, j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            up_score = score[i - 1, j] + gap
            left_score = score[i, j - 1] + gap
            score[i, j] = max(0, diag_score, up_score, left_score)

            if score[i, j] == 0:
                traceback[i, j] = -1  # Indicates start of alignment
            elif score[i, j] == diag_score:
                traceback[i, j] = 0
            elif score[i, j] == up_score:
                traceback[i, j] = 1
            else:
                traceback[i, j] = 2

    return score, traceback

def find_traceback(score, traceback, seq1, seq2):
    """
    Perform iterative traceback to find the best local alignment path.
    Returns aligned strings, score, relative start and end indices.
    """
    if score.size == 0 or np.max(score) == 0:
        return "", "", 0, 0, 0, 0, 0

    i, j = np.unravel_index(np.argmax(score), score.shape)
    max_score = score[i, j]

    if max_score == 0:
        return "", "", 0, 0, 0, 0, 0

    align1 = []
    align2 = []
    curr_i, curr_j = i, j
    end1_rel = i - 1
    end2_rel = j - 1

    while curr_i > 0 and curr_j > 0 and score[curr_i, curr_j] > 0:
        direction = traceback[curr_i, curr_j]
        if direction == 0:  # Diagonal match/mismatch
            align1.append(seq1[curr_i - 1])
            align2.append(seq2[curr_j - 1])
            curr_i -= 1
            curr_j -= 1
        elif direction == 1:  # Up: insertion in seq1 (gap in seq2)
            align1.append(seq1[curr_i - 1])
            align2.append('-')
            curr_i -= 1
        elif direction == 2:  # Left: deletion in seq1 (gap in seq1)
            align1.append('-')
            align2.append(seq2[curr_j - 1])
            curr_j -= 1
        else:
            break  # Should not occur

    start1_rel = curr_i
    start2_rel = curr_j

    align1.reverse()
    align2.reverse()

    align1_str = ''.join(align1)
    align2_str = ''.join(align2)

    return align1_str, align2_str, max_score, start1_rel, start2_rel, end1_rel, end2_rel

def multi_homolog_smith_waterman_alg(seq1, seq2, match=2, mismatch=-1, gap=-1, min_score_threshold=5):
    """
    Improved multi-homologous Smith-Waterman algorithm for detecting and aligning
    multiple non-overlapping homologous regions symmetrically in both sequences.
    Handles splitting, overlap prevention, dynamic iterations, and validation.
    """
    if not seq1 or not seq2:
        return []

    # Maintain lists of unaligned segments as (start, end) inclusive indices
    seq1_segments = [(0, len(seq1) - 1)]
    seq2_segments = [(0, len(seq2) - 1)]

    alignments = []

    while seq1_segments and seq2_segments:
        best_score = 0
        best_align1 = ""
        best_align2 = ""
        best_start1_abs = 0
        best_end1_abs = 0
        best_start2_abs = 0
        best_end2_abs = 0
        best_i_seg = -1
        best_j_seg = -1

        # Compute alignments for all pairs of current segments
        for i_seg, (s1, e1) in enumerate(seq1_segments):
            if s1 > e1:
                continue  # Skip invalid segments
            sub1 = seq1[s1:e1 + 1]
            for j_seg, (s2, e2) in enumerate(seq2_segments):
                if s2 > e2:
                    continue
                sub2 = seq2[s2:e2 + 1]
                score_mat, tb_mat = score_traceback_matrixs(sub1, sub2, match, mismatch, gap)
                a1, a2, sc, st1_rel, st2_rel, en1_rel, en2_rel = find_traceback(score_mat, tb_mat, sub1, sub2)
                if sc > best_score and sc >= min_score_threshold:
                    best_score = sc
                    best_align1 = a1
                    best_align2 = a2
                    best_start1_abs = s1 + st1_rel
                    best_end1_abs = s1 + en1_rel
                    best_start2_abs = s2 + st2_rel
                    best_end2_abs = s2 + en2_rel
                    best_i_seg = i_seg
                    best_j_seg = j_seg

        if best_score < min_score_threshold:
            break  # No significant alignments left

        # Store the alignment
        alignment = {
            'aligns': (best_align1, best_align2),
            'start_index': (best_start1_abs, best_start2_abs),
            'end_index': (best_end1_abs, best_end2_abs),
            'score': best_score
        }
        alignments.append(alignment)

        # Print summary
        print(f"Homologous region {len(alignments)}:")
        print(f"Score: {best_score}")
        print(f"Seq1 [{best_start1_abs}:{best_end1_abs + 1}]: {best_align1}")
        print(f"Seq2 [{best_start2_abs}:{best_end2_abs + 1}]: {best_align2}")
        print()

        # Split seq1 segment symmetrically around the aligned region
        s1, e1 = seq1_segments[best_i_seg]
        pre1_s = s1
        pre1_e = best_start1_abs - 1
        post1_s = best_end1_abs + 1
        post1_e = e1

        # Remove the processed segment
        del seq1_segments[best_i_seg]

        # Add pre and post if non-empty
        if pre1_s <= pre1_e:
            seq1_segments.append((pre1_s, pre1_e))
        if post1_s <= post1_e:
            seq1_segments.append((post1_s, post1_e))

        # Split seq2 segment symmetrically
        s2, e2 = seq2_segments[best_j_seg]
        pre2_s = s2
        pre2_e = best_start2_abs - 1
        post2_s = best_end2_abs + 1
        post2_e = e2

        del seq2_segments[best_j_seg]

        if pre2_s <= pre2_e:
            seq2_segments.append((pre2_s, pre2_e))
        if post2_s <= post2_e:
            seq2_segments.append((post2_s, post2_e))

        # Sort segments by start index to maintain order
        seq1_segments.sort(key=lambda x: x[0])
        seq2_segments.sort(key=lambda x: x[0])

    return alignments

# Test section using sample sequences designed to demonstrate multiple homologous regions
if __name__ == "__main__":
    # Sample sequences with multiple non-overlapping homologous regions
    # seq1 has two regions matching parts of seq2, separated by non-homologous parts
    seq1 = "ATGCATGCXXATGCATGCTAG"
    seq2 = "TGCATGCTGCATGYYY"

    print("Running multi-homologous Smith-Waterman alignment...")
    results = multi_homolog_smith_waterman_alg(seq1, seq2)
    print("All alignments found:")
    for idx, res in enumerate(results, 1):
        aligns = res['aligns']
        starts = res['start_index']
        ends = res['end_index']
        score = res['score']
        print(f"Alignment {idx}: Score={score}, Seq1[{starts[0]}:{ends[0]+1}]={aligns[0]}, Seq2[{starts[1]}:{ends[1]+1}]={aligns[1]}")