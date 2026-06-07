# R Code Standards

## Environment

- **Package Manager:** `renv` — use `renv::restore()`, `renv::snapshot()`
- **Package Loading:** `pacman::p_load()` at script start
- **Line Length:** 80 characters

## Style

- Inside reusable functions, prefer explicit namespaces for non-base calls.
- Preserve existing R style. All changes should look as if written by the same person who wrote the existing code, unless differently specified.
- Do not introduce idioms absent from the existing scripts unless no equivalent exists, but justify the choice explicitly.

## Pipe Syntax

**Use native pipe `|>` exclusively.** Never `%>%`.

```r
# Correct
result <- data |> 
  filter(!is.na(value)) |> 
  summarise(mean_value = mean(value))

# Wrong
result <- data %>% 
  filter(!is.na(value))
```

## Syntax Approach

**Hybrid:** tidyverse for cleaning, data.table for performance.

```r
# Cleaning with tidyverse
data <- raw_data |>
  mutate(date = ymd(date_string)) |>
  filter(year >= 1780)

# Performance with data.table
results <- as.data.table(data)[
  , .(total = sum(value)), 
  by = .(region, year)
]
```

## Path Handling

**Use `here::here()` for robust relative paths.**

```r
library(here)

raw_data <- here("data", "raw", "mortality.csv")
output <- here("out", "tables", "results.csv")
```

## Documentation

**roxygen2 comments** for functions:

```r
#' Calculate mortality rate by region
#'
#' @description Computes crude death rates from parish records
#' @param data Data frame with deaths and population columns
#' @param year_col Name of the year column
#' @return Data frame with mortality_rate column
#' @export
calculate_mortality <- function(data, year_col = "year") {
  # Implementation
}
```

## Imports

- Load at top: `pacman::p_load()` or `library()`
- **Always** use explicit namespaces in function bodies: `dplyr::mutate()`, `data.table::as.data.table()`

## Naming

| Type | Convention |
|------|-----------|
| Variables/Functions | `snake_case` |
| Constants | `UPPER_SNAKE_CASE` |
| Files | Descriptive names with `.R` extension |

## Commands

```r
# Restore environment
renv::restore()

# Install package
renv::install("dplyr")

# Snapshot changes
renv::snapshot()

# Load packages
pacman::p_load(dplyr, data.table, here, sf)
```

## Error Handling

Use `tryCatch()` for all external operations (file I/O, API calls, downloads).

## I/O Operations

- Use `rio` for I/O operations where possible

```r
# To read data
data <- rio::import(...)
# To write data
rio::export(data, here(...))
```
