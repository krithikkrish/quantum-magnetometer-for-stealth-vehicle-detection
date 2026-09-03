module hil_top (
    input  wire clk,
    input  wire rst,
    input  wire uart_rx_pin,
    output wire uart_tx_pin,
    output wire mw_gate_pin,
    output wire pwm_pin 
);

    wire [15:0] current_tau;
    wire        tau_valid;
    wire [7:0]  snr_val;
    wire        snr_ready;
    wire        cpmg_done;
    
    // DE10-Standard KEYs are active-low. Invert it so rst_active is high when pressed.
    // If you don't press the button, rst_active is low, and the system runs normally!
    wire        rst_active = ~rst;
    
    reg         cpmg_start;
    reg         tx_req;
    wire        tx_busy;

    assign pwm_pin = 1'b0; 

    uart_rx rx_inst (.clk(clk), .rst(rst_active), .rx_pin(uart_rx_pin), .tau_out(current_tau), .tau_valid(tau_valid));
    
    cpmg_sequencer cpmg_inst (
        .clk(clk), .rst(rst_active), .start(cpmg_start), .tau(current_tau), 
        .n_pulses(8'd4), // Set to 4 for fast simulation
        .mw_gate(mw_gate_pin), .done(cpmg_done)
    );

    snr_estimator snr_inst (.clk(clk), .rst(rst_active), .trigger(cpmg_done), .tau(current_tau), .snr_out(snr_val), .data_ready(snr_ready));

    uart_tx tx_inst (.clk(clk), .rst(rst_active), .tx_req(tx_req), .tx_data(snr_val), .tx_pin(uart_tx_pin), .busy(tx_busy));

    reg [1:0] state;
    always @(posedge clk or posedge rst_active) begin
        if (rst_active) begin
            state      <= 2'd0;
            cpmg_start <= 1'b0;
            tx_req     <= 1'b0;
        end else begin
            cpmg_start <= 1'b0;
            tx_req     <= 1'b0;
            
            case (state)
                2'd0: begin
                    cpmg_start <= 1'b1;
                    state      <= 2'd1;
                end
                2'd1: begin
                    if (snr_ready) begin
                        tx_req <= 1'b1; 
                        state  <= 2'd2;
                    end
                end
                2'd2: begin
                    if (!tx_busy) state <= 2'd0; 
                end
            endcase
        end
    end
endmodule
