import re

YOUTRACK_ISSUE = re.compile("[A-Z]+-[0-9]+")


def is_empty_comment(comment: str) -> bool:
    comment = comment.strip()
    return (comment == "") or (comment == "Empty")


def is_auto_marker_comment(comment: str) -> bool:
    # Comment is from not reviewed auto-marker issue.
    comment = comment.lower()
    return "auto-marker" in comment


def is_only_youtrack_issue_comment(comment: str) -> bool:
    comment = comment.strip()
    return YOUTRACK_ISSUE.fullmatch(comment) is not None


def is_unsorted_comment(comment: str) -> bool:
    comment = comment.strip()
    return "unsorted" in comment


def is_sorted_comment(comment: str) -> bool:
    predicates = [
        not is_empty_comment(comment),
        not is_auto_marker_comment(comment),
        not is_only_youtrack_issue_comment(comment),
        not is_unsorted_comment(comment)
    ]
    return all(predicates)
