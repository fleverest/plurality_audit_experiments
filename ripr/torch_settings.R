box::use(torch[cuda_is_available, torch_device, torch_float64])

#' Active torch device: CUDA if available, otherwise CPU
#' @export
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

#' Default tensor dtype: float64
#' @export
dtype <- torch_float64()
