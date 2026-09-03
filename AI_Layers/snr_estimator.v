module snr_estimator (
    input  wire        clk,
    input  wire        rst,
    input  wire        trigger,  // Triggered when CPMG sequence is done
    input  wire [15:0] tau,      // Current tau being tested
    output reg  [7:0]  snr_out,  // 0-255 code to send to PC
    output reg         data_ready
);

    // Target resonant tau = 500 cycles (10 us at 50 MHz for 25 kHz signal)
    localparam TARGET_TAU = 16'd500;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            snr_out    <= 8'd0;
            data_ready <= 1'b0;
        end else begin
            data_ready <= 1'b0; 
            
            if (trigger) begin
                if (tau >= TARGET_TAU) begin
                    if ((tau - TARGET_TAU) < 16'd127)
                        // Removed the illegal bit-slice. Verilog truncates this automatically.
                        snr_out <= 8'd255 - (tau - TARGET_TAU); 
                    else
                        snr_out <= 8'd10; 
                end else begin
                    if ((TARGET_TAU - tau) < 16'd127)
                        // Removed the illegal bit-slice.
                        snr_out <= 8'd255 - (TARGET_TAU - tau);
                    else
                        snr_out <= 8'd10; 
                end
                data_ready <= 1'b1; 
            end
        end
    end
endmodule
