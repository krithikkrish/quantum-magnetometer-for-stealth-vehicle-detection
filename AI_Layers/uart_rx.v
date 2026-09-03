module uart_rx (
    input  wire        clk,
    input  wire        rst,
    input  wire        rx_pin,
    output reg  [15:0] tau_out,
    output reg         tau_valid
);

    localparam CLKS_PER_BIT = 16'd434;
    
    reg [15:0] clk_count;
    reg [3:0]  bit_idx;
    reg [7:0]  rx_byte;
    reg        receiving;
    reg        byte_count; 
    reg [7:0]  high_byte_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tau_out    <= 16'd500; 
            tau_valid  <= 1'b0;
            receiving  <= 1'b0;
            clk_count  <= 16'd0;
            bit_idx    <= 4'd0;
            byte_count <= 1'b0;
        end else begin
            tau_valid <= 1'b0; 
            
            if (!receiving && rx_pin == 1'b0) begin
                receiving <= 1'b1;
                // Wait 1.5 bit periods to reach the center of Data Bit 0!
                clk_count <= CLKS_PER_BIT + (CLKS_PER_BIT / 2); 
                bit_idx   <= 4'd0;
            end else if (receiving) begin
                if (clk_count > 0) begin
                    clk_count <= clk_count - 16'd1;
                end else begin
                    clk_count <= CLKS_PER_BIT;
                    
                    if (bit_idx < 4'd8) begin
                        rx_byte[bit_idx] <= rx_pin;
                        bit_idx <= bit_idx + 4'd1;
                    end else begin
                        receiving <= 1'b0;
                        if (byte_count == 1'b0) begin
                            high_byte_reg <= rx_byte;
                            byte_count    <= 1'b1; 
                        end else begin
                            tau_out    <= {high_byte_reg, rx_byte};
                            tau_valid  <= 1'b1;
                            byte_count <= 1'b0; 
                        end
                    end
                end
            end
        end
    end
endmodule
