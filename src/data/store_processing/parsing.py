from typing import Dict, List, Union

from pydantic import BaseModel, Field, validator


class PriceModel(BaseModel):
    code: str = None
    fractional: int = None
    formatted: str = None


class ImageModel(BaseModel):
    typeName: str = None
    altText: str = None
    url: str = None
    type: str = None


class ItemModel(BaseModel):
    percentageDiscounted: Union[str, None]
    name: str
    popular: bool
    categoryId: str
    modifierGroupIds: List
    priceDiscounted: Union[PriceModel, None] = PriceModel()
    isSignatureExclusive: bool
    productInformation: str
    price: PriceModel = Field(...)
    nutritionalInfo: Union[str, None]
    available: bool
    maxSelection: Union[str, None]
    description: Union[str, None]
    image: Union[ImageModel, None]
    id: str

    @validator("priceDiscounted")
    def check_price_object(cls, v):
        if not v:
            return PriceModel()
        else:
            return v

    @validator("image")
    def check_image_object(cls, v):
        if not v:
            return ImageModel()
        else:
            return v


class AddressModel(BaseModel):
    address1: str
    postCode: Union[str, None]
    neighborhood: str
    city: str
    country: str


class LocationModel(BaseModel):
    cityId: int
    zoneId: int
    address: AddressModel = Field(...)


class RestaurantModel(BaseModel):
    id: str
    name: str
    hasOrderNotes: bool
    tipMessage: Union[str, None]
    menuDisabled: bool
    deliversToCustomerLocation: bool
    menuId: str
    drnId: str
    currencyCode: str
    currencySymbol: str
    location: LocationModel = Field(...)


class MinimumOrderValueModel(BaseModel):
    code: str
    fractional: int
    formatted: str


class InnerOfferModel(BaseModel):
    minimumOrderValue: MinimumOrderValueModel = Field(...)


class OfferModel(BaseModel):
    offer: InnerOfferModel = Field(...)
    progressBar: Dict


class MetaModel(BaseModel):
    restaurant: RestaurantModel = Field(...)
    items: List[ItemModel] = Field(...)
    metatags: Dict
    requestUUID: str
    appliedParams: List
    customerLocation: Dict
    offer: Union[OfferModel, None]
    modifierGroups: List
    categories: List
    pastOrders: List


class MenuModel(BaseModel):
    meta: MetaModel = Field(...)
    layoutGroups: List
    header: Dict
    navigationGroups: Dict
    modals: List
    loading: bool
    errors: Dict
    menuBanners: List
    footerBanner: Union[Dict, None]
    actionModal: Union[Dict, None]
    basketBlockConfirmation: Union[Dict, None]
    capabilities: List


class MenuPageModel(BaseModel):
    menu: MenuModel = Field(...)
    basket: Dict
    actionModal: Union[Dict, None]
    search: Dict


class InitialStateModel(BaseModel):
    menuPage: MenuPageModel = Field(...)
    account: Dict
    apiRequests: Dict
    basket: Dict
    brands: Dict
    checkoutPayment: Dict
    config: Dict
    creditRedemption: Dict
    cuisine: Dict
    favourites: Dict
    home: Dict
    homepageFeaturedContent: Dict
    landingPage: Dict
    location: Dict
    mealCard: Dict
    notification: Dict
    order: Dict
    referral: Dict
    request: Dict
    staticPage: Dict
    subscriptions: Dict
    user: Dict
    voteArea: Dict
    voucher: Dict
    orderRating: Dict
    orderReviews: Dict


class PropsModel(BaseModel):
    initialState: InitialStateModel = Field(...)
    translations: Dict
    additionalProps: Dict
    sentryContext: Dict
    showLoginSuccessBanner: bool


class DataModel(BaseModel):
    props: PropsModel = Field(...)
    page: str
    query: Dict
    buildId: str
    assetPrefix: str
    runtimeConfig: Dict
    isFallback: bool
    dynamicIds: List
    customServer: bool
    appGip: bool
    scriptLoader: List


def structure_data(raw_data: List) -> DataModel:
    data = raw_data[0]["data"]
    return DataModel(**data)


def get_store_data(data: DataModel) -> RestaurantModel:
    return data.props.initialState.menuPage.menu.meta.restaurant


def get_menu_data(data: DataModel) -> List[ItemModel]:
    return data.props.initialState.menuPage.menu.meta.items


def transform_data_store(raw_data: List) -> List[Dict]:
    model_data = structure_data(raw_data)
    store_data = get_store_data(model_data)
    return [store_data.dict()]


def transform_data_menu(raw_data: List) -> List[Dict]:
    model_data = structure_data(raw_data)
    menu_data = get_menu_data(model_data)
    return [el.dict() for el in menu_data]
