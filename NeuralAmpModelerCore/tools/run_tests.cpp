// Entry point for tests
// See the GitHub Action for a demo how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_a2_container.cpp"
#include "test/test_conv1d.cpp"
#include "test/test_conv_1x1.cpp"
#include "test/test_convnet.cpp"
#include "test/test_dsp.cpp"
#include "test/test_film.cpp"
#include "test/test_film_realtime_safe.cpp"
#include "test/test_fast_lut.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_ring_buffer.cpp"
#include "test/test_wavenet/test_layer.cpp"
#include "test/test_wavenet/test_layer_array.cpp"
#include "test/test_wavenet/test_full.cpp"
#include "test/test_wavenet/test_real_time_safe.cpp"
#include "test/test_wavenet/test_condition_processing.cpp"
#include "test/test_wavenet/test_head1x1.cpp"
#include "test/test_gating_activations.cpp"
#include "test/test_wavenet_gating_compatibility.cpp"
#include "test/test_blending_detailed.cpp"
#include "test/test_input_buffer_verification.cpp"
#include "test/test_lstm.cpp"
#include "test/test_player_manifest.cpp"
#include "test/test_wavenet_configurable_gating.cpp"

int main()
{
  std::cout << "Running tests..." << std::endl;
  // TODO Automatically loop, catch exceptions, log results

  test_activations::TestFastTanh::test_core_function();

  test_a2_container::test_container_defaults_to_full_and_switches();
  test_a2_container::test_a2_detector_accepts_bounded_global_conditioning();
  test_a2_container::test_a2_detector_rejects_integer_wraparound();
  test_a2_container::test_conditioned_a2_fast_kernel_uses_controls_and_defaults();
  test_a2_container::test_a2_detector_rejects_unsupported_variants();
  test_player_manifest::test_dess_package_is_self_contained_and_loadable();
  test_player_manifest::test_custom_import_keeps_verified_packages_after_source_moves();
  test_player_manifest::test_player_dsp_revalidates_the_bytes_it_parses();
  test_player_manifest::test_package_manifest_parser_binds_the_verified_cache_bytes();
  test_player_manifest::test_dess_package_rejects_corruption_and_trailing_data();
  test_player_manifest::test_morph_package_loads_both_models_and_renders();
  test_player_manifest::test_morph_blend_law_endpoints_center_and_realtime_safety();
  test_player_manifest::test_condition_control_display_range_preserves_model_mapping();
  test_player_manifest::test_condition_control_display_range_parser();
  test_player_manifest::test_condition_control_taper_parser();
  test_player_manifest::test_fixed_switch_control_parser();
  test_player_manifest::test_presentation_head_shell_parser();
  test_activations::TestFastTanh::test_get_by_init();
  test_activations::TestFastTanh::test_get_by_str();

  test_activations::TestLeakyReLU::test_core_function();
  test_activations::TestLeakyReLU::test_get_by_init();
  test_activations::TestLeakyReLU::test_get_by_str();

  test_lut::TestFastLUT::test_sigmoid();
  test_lut::TestFastLUT::test_tanh();

  test_activations::TestPReLU::test_core_function();
  test_activations::TestPReLU::test_per_channel_behavior();
  test_activations::TestPReLU::test_single_slope_broadcast();
  test_activations::TestPReLU::test_wrong_number_of_channels();

  // Typed ActivationConfig tests
  test_activations::TestTypedActivationConfig::test_simple_config();
  test_activations::TestTypedActivationConfig::test_all_simple_types();
  test_activations::TestTypedActivationConfig::test_leaky_relu_config();
  test_activations::TestTypedActivationConfig::test_prelu_single_slope_config();
  test_activations::TestTypedActivationConfig::test_prelu_multi_slope_config();
  test_activations::TestTypedActivationConfig::test_leaky_hardtanh_config();
  test_activations::TestTypedActivationConfig::test_from_json_string();
  test_activations::TestTypedActivationConfig::test_from_json_object();
  test_activations::TestTypedActivationConfig::test_from_json_prelu_multi();
  test_activations::TestTypedActivationConfig::test_rejects_excessive_or_nonfinite_prelu_slopes();
  test_activations::TestTypedActivationConfig::test_unknown_activation_throws();

  test_dsp::test_construct();
  test_dsp::test_get_input_level();
  test_dsp::test_get_output_level();
  test_dsp::test_has_input_level();
  test_dsp::test_has_output_level();
  test_dsp::test_set_input_level();
  test_dsp::test_set_output_level();
  test_dsp::test_audio_metadata_rejects_unsafe_values();
  test_dsp::test_prewarm_processes_only_requested_frames();

  test_ring_buffer::test_construct();
  test_ring_buffer::test_reset();
  test_ring_buffer::test_reset_with_receptive_field();
  test_ring_buffer::test_write();
  test_ring_buffer::test_read_with_lookback();
  test_ring_buffer::test_advance();
  test_ring_buffer::test_rewind();
  test_ring_buffer::test_multiple_writes_reads();
  test_ring_buffer::test_reset_zeros_history_area();
  test_ring_buffer::test_rewind_preserves_history();

  test_conv1d::test_construct();
  test_conv1d::test_construct_rejects_zero_groups();
  test_conv1d::test_construct_rejects_dense_grouped_amplification();
  test_conv1d::test_set_size();
  test_conv1d::test_reset();
  test_conv1d::test_process_basic();
  test_conv1d::test_process_with_bias();
  test_conv1d::test_process_multichannel();
  test_conv1d::test_process_dilation();
  test_conv1d::test_process_multiple_calls();
  test_conv1d::test_get_output_different_sizes();
  test_conv1d::test_set_size_and_weights();
  test_conv1d::test_get_num_weights();
  test_conv1d::test_reset_multiple();
  test_conv1d::test_process_grouped_basic();
  test_conv1d::test_process_grouped_with_bias();
  test_conv1d::test_process_grouped_multiple_groups();
  test_conv1d::test_process_grouped_kernel_size();
  test_conv1d::test_process_grouped_dilation();
  test_conv1d::test_process_grouped_channel_isolation();
  test_conv1d::test_get_num_weights_grouped();

  test_conv_1x1::test_construct();
  test_conv_1x1::test_construct_with_groups();
  test_conv_1x1::test_construct_validation_zero_groups();
  test_conv_1x1::test_construct_validation_dense_grouped_amplification();
  test_conv_1x1::test_construct_validation_in_channels();
  test_conv_1x1::test_construct_validation_out_channels();
  test_conv_1x1::test_process_basic();
  test_conv_1x1::test_process_with_bias();
  test_conv_1x1::test_process_underscore();
  test_conv_1x1::test_process_grouped_basic();
  test_conv_1x1::test_process_grouped_with_bias();
  test_conv_1x1::test_process_grouped_multiple_groups();
  test_conv_1x1::test_process_grouped_channel_isolation();
  test_conv_1x1::test_process_underscore_grouped();
  test_conv_1x1::test_set_max_buffer_size();
  test_conv_1x1::test_process_multiple_calls();

  test_film::test_set_max_buffer_size();
  test_film::test_process_bias_only();
  test_film::test_process_scale_only();
  test_film::test_process_inplace_with_shift();
  test_film::test_process_inplace_scale_only();
  test_film::test_process_inplace_partial_frames();

  test_film_realtime_safe::test_allocation_tracking_pass();
  test_film_realtime_safe::test_allocation_tracking_fail();
  test_film_realtime_safe::test_film_process_with_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_without_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_inplace_with_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_inplace_without_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_large_dimensions_realtime_safe();
  test_film_realtime_safe::test_film_process_partial_frames_realtime_safe();
  test_film_realtime_safe::test_film_process_varying_dimensions_realtime_safe();
  test_film_realtime_safe::test_film_process_consecutive_calls_realtime_safe();

  test_wavenet::test_layer::test_gated();
  test_wavenet::test_layer::test_layer_getters();
  test_wavenet::test_layer::test_non_gated_layer();
  test_wavenet::test_layer::test_layer_activations();
  test_wavenet::test_layer::test_layer_multichannel();
  test_wavenet::test_layer::test_layer_bottleneck();
  test_wavenet::test_layer::test_layer_bottleneck_gated();
  test_wavenet::test_layer_array::test_layer_array_basic();
  test_wavenet::test_layer_array::test_layer_array_receptive_field();
  test_wavenet::test_layer_array::test_layer_array_with_head_input();
  test_wavenet::test_full::test_wavenet_model();
  test_wavenet::test_full::test_wavenet_multiple_arrays();
  test_wavenet::test_full::test_wavenet_zero_input();
  test_wavenet::test_full::test_wavenet_different_buffer_sizes();
  test_wavenet::test_full::test_wavenet_prewarm();
  test_wavenet::test_full::test_wavenet_exact_weight_count_includes_all_films();
  test_wavenet::test_full::test_wavenet_rejects_short_and_surplus_weights();
  test_wavenet::test_full::test_wavenet_rejects_invalid_groups_dilations_and_gating();
  test_wavenet::test_full::test_wavenet_rejects_input_condition_and_head_chain_mismatches();
  test_wavenet::test_full::test_wavenet_rejects_head_film_without_head_projection();
  test_wavenet::test_full::test_wavenet_counts_nested_linear_prewarm_compute();
  test_wavenet::test_head1x1::test_head1x1_inactive();
  test_wavenet::test_head1x1::test_head1x1_active();
  test_wavenet::test_head1x1::test_head1x1_gated();
  test_wavenet::test_head1x1::test_head1x1_groups();
  test_wavenet::test_head1x1::test_head1x1_different_out_channels();
  test_wavenet::test_allocation_tracking_pass();
  test_wavenet::test_allocation_tracking_fail();
  test_wavenet::test_conv1d_process_realtime_safe();
  test_wavenet::test_conv1d_grouped_process_realtime_safe();
  test_wavenet::test_conv1d_grouped_dilated_process_realtime_safe();
  test_wavenet::test_layer_process_realtime_safe();
  test_wavenet::test_layer_bottleneck_process_realtime_safe();
  test_wavenet::test_layer_grouped_process_realtime_safe();
  test_wavenet::test_layer_all_films_with_shift_realtime_safe();
  test_wavenet::test_layer_all_films_without_shift_realtime_safe();
  test_wavenet::test_layer_post_activation_film_gated_realtime_safe();
  test_wavenet::test_layer_post_activation_film_blended_realtime_safe();
  test_wavenet::test_layer_array_process_realtime_safe();
  test_wavenet::test_process_realtime_safe();
  test_wavenet::test_process_3in_2out_realtime_safe();
  test_wavenet::test_condition_processing::test_with_condition_dsp();
  test_wavenet::test_condition_processing::test_with_condition_dsp_multichannel();
  test_wavenet::test_condition_processing::test_with_condition_dsp_and_global_condition();

  test_convnet::test_convnet_basic();
  test_convnet::test_convnet_batchnorm();
  test_convnet::test_convnet_multiple_blocks();
  test_convnet::test_convnet_zero_input();
  test_convnet::test_convnet_different_buffer_sizes();
  test_convnet::test_convnet_prewarm();
  test_convnet::test_convnet_multiple_calls();
  test_convnet::test_convnet_grouped_weight_count();
  test_convnet::test_convnet_rejects_short_and_surplus_weights();
  test_convnet::test_convnet_rejects_malformed_dimensions();

  // LSTM tests
  test_lstm::test_lstm_basic();
  test_lstm::test_lstm_multiple_layers();
  test_lstm::test_lstm_zero_input();
  test_lstm::test_lstm_different_buffer_sizes();
  test_lstm::test_lstm_prewarm();
  test_lstm::test_lstm_multiple_calls();
  test_lstm::test_lstm_multichannel();
  test_lstm::test_lstm_large_hidden_size();
  test_lstm::test_lstm_different_input_size();
  test_lstm::test_lstm_state_evolution();
  test_lstm::test_lstm_no_layers();
  test_lstm::test_lstm_input_size_can_exceed_channel_count();
  test_lstm::test_lstm_rejects_short_and_surplus_weights();
  test_lstm::test_lstm_rejects_malformed_dimensions();

  // Gating activations tests
  test_gating_activations::TestGatingActivation::test_basic_functionality();
  test_gating_activations::TestGatingActivation::test_with_custom_activations();
  //  test_gating_activations::TestGatingActivation::test_error_handling();

  // Wavenet gating compatibility tests
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_wavenet_style_gating();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_column_by_column_processing();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_memory_contiguity();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_multiple_channels();

  test_gating_activations::TestBlendingActivation::test_basic_functionality();
  test_gating_activations::TestBlendingActivation::test_blending_behavior();
  test_gating_activations::TestBlendingActivation::test_with_custom_activations();
  //  test_gating_activations::TestBlendingActivation::test_error_handling();
  test_gating_activations::TestBlendingActivation::test_edge_cases();

  // Detailed blending tests
  test_blending_detailed::TestBlendingDetailed::test_blending_with_different_activations();
  test_blending_detailed::TestBlendingDetailed::test_input_buffer_usage();

  // Input buffer verification tests
  test_input_buffer_verification::TestInputBufferVerification::test_buffer_stores_pre_activation_values();
  test_input_buffer_verification::TestInputBufferVerification::test_buffer_with_different_activations();

  // Configurable gating/blending tests
  run_configurable_gating_tests();

  test_get_dsp::test_gets_input_level();
  test_get_dsp::test_gets_output_level();
  test_get_dsp::test_null_input_level();
  test_get_dsp::test_null_output_level();
  test_get_dsp::test_missing_metadata_is_optional();
  test_get_dsp::test_missing_required_top_level_field_throws();
  test_get_dsp::test_rejects_malformed_or_unsupported_versions();
  test_get_dsp::test_rejects_unsafe_numeric_metadata();
  test_get_dsp::test_null_sample_rate_is_unknown();
  test_get_dsp::test_rejects_non_integer_architecture_dimensions();
  test_get_dsp::test_rejects_unsafe_linear_resources_before_allocation();
  test_get_dsp::test_rejects_unsafe_aggregate_prewarm_buffers();
  test_get_dsp::test_rejects_excessive_lstm_depth();
  test_get_dsp::test_rejects_excessive_convolution_prewarm_work();
  test_get_dsp::test_rejects_width_amplified_prewarm_compute();
  test_get_dsp::test_rejects_nonfinite_weights_and_invalid_metadata_shape();
  test_get_dsp::test_rejects_nonfinite_direct_dsp_data_weights();
  test_get_dsp::test_rejects_excessive_model_weight_count_before_reserve();
  test_get_dsp::test_rejects_aggregate_nested_model_weights();
  test_get_dsp::test_rejects_excessive_aggregate_model_count();
  test_get_dsp::test_rejects_excessive_config_and_metadata_json_budgets();
  test_get_dsp::test_rejects_excessive_nested_condition_models();
  test_get_dsp::test_rejects_mismatched_prelu_channels();
  test_get_dsp::test_rejects_oversized_model_file_before_parsing();
  test_get_dsp::test_rejects_parse_time_json_event_amplification();
  test_get_dsp::test_rejects_parse_time_structural_node_amplification();

  // Finally, some end-to-end tests.
  test_get_dsp::test_load_and_process_nam_files();

  std::cout << "Success!" << std::endl;
#ifdef ADDASSERT
  std::cerr << "===============================================================" << std::endl;
  std::cerr << "Checking that we're successfully asserting. We should now fail." << std::endl;
  assert(false);
#endif
  return 0;
}
