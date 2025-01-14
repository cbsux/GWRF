import setuptools

with open("README_PYTHON.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GWRF",
    version="1.0",
    author="ChiBeiSheng",
    url='xxx',
    author_email="cbs3307821258@qq.com",
    description="Geographically Weighted Random Forest",
    long_description="GWRF (Geographically Weighted Random Forest) is a Python library designed to incorporate the "
                     "concept of geographical weighting into the random forest model, enabling regression analysis "
                     "of data with spatial correlation. It integrates powerful tools such as numpy, xgboost, and sklearn, "
                     "providing users with a range of functionalities from model training and prediction to feature "
                     "importance assessment and partial dependence analysis, while also supporting model saving and loading",
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
