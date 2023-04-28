from misc_utils import *

def maybe_inline_expr(ctor, expr):
    if expr[0] == ctor:
        return expr[1]
    else:
        return (expr,)

def fill_i_j(enumerated, i, j):
    assert (i, j) not in enumerated

    new_set = set()

    for (i_l, i_r), (j_l, j_r) in integer_tuple_partition(2, (i, j)):
        if (i_l == i and j_l == j) or (i_r == i and j_r == j):
            # if we're trying to a build thing of size (i, j), in principle we could use
            # a thing of size (i, j) and a thing of size (0, 0), but there are no things
            # of size (0, 0), and yet if we didn't bail here we'd look at enumerated[i, j]
            # and find it doesn't exist (since we're trying to populate it!)
            continue
        
        for l in enumerated[i_l, j_l]:
            for r in enumerated[i_r, j_r]:
                new_set.add(
                    ("CONJ", (
                        *maybe_inline_expr("CONJ", l),
                        *maybe_inline_expr("CONJ", r),
                    ))
                )
                # FIXME: this is probably leading to redundancy in later steps
                # by only doing it in one order we get rid of commutativity
                # duplication, but not associativity duplication
                # taking a 1 and a 1 is okay, but if we take a 1 and a 2 or a
                # 2 and a 1, there's overlap
                # FIXME: wait I think the proceeding is optimistic: we enumerate the
                # big sets once on the left and once on the right, so we are also losing
                # out on commutativity

                new_set.add(
                    ("SEQ", (
                        *maybe_inline_expr("SEQ", l),
                        *maybe_inline_expr("SEQ", r),
                    ))
                )
                new_set.add(
                    ("SEQ", (
                        *maybe_inline_expr("SEQ", r),
                        *maybe_inline_expr("SEQ", l),
                    ))
                )
    
    enumerated[i, j] = frozenset(new_set)

def fill_up_to(enumerated, n, m):
    for i in range(n + 1):
        for j in range(m + 1):
            if (i, j) not in enumerated:
                fill_i_j(enumerated, i, j)

def enumerate_up_to_separate_01(pred0, pred1, num_pred0, num_pred1):
    enumerated = {
        (0, 0): frozenset(),
        (1, 0): pred0,
        (0, 1): pred1,
    }
    fill_up_to(enumerated, num_pred0, num_pred1)
    return frozenset(x for k, v in enumerated.items() for x in v)

def enumerate_up_to_combined_01(pred0, pred1, num_pred01):
    enumerated = {
        (0, 0): frozenset(),
        (1, 0): pred0,
        (0, 1): pred1,
    }
    for i in range(num_pred01 + 1):
        fill_up_to(enumerated, i, num_pred01 - i)
    return frozenset(x for k, v in enumerated.items() for x in v)

if False:
    len(enumerate_up_to_separate_01(
        frozenset({
            ("PRED0", "top"),
            ("PRED0", "bot"),
        }),
        frozenset({
            ("PRED1", "t+", None),
            ("PRED1", "t-", None),
        }),
        2, 1,
    ))