import pandas as pd


country_list = pd.read_csv("../data/country_list.csv")


def map_id_to_country(country_id: int, mode: str) -> str:
    if mode == "gw":
        return country_list[country_list["country_id"] == country_id].name.reset_index(
            drop=True
        )[0]
    if mode == "ccode":
        raise NotImplementedError("ccode mode not implemented yet")


def map_country_to_id(country: str, mode: str) -> int:
    if mode == "gw":
        return country_list[country_list["name"] == country].country_id.reset_index(
            drop=True
        )[0]
    if mode == "ccode":
        raise NotImplementedError("ccode mode not implemented yet")
