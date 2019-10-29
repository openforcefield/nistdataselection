"""
Unit test for the nistdataselection package.
"""


def test_nistdataselection_imported():
    """Sample test, will always pass so long as import statement worked"""
    from nistdataselection import curation, processing, reporting

    assert curation is not None
    assert processing is not None
    assert reporting is not None
