module uart_tx (
    input  wire       clk,
    input  wire       rst,
    input  wire       tx_req,
    input  wire [7:0] tx_data,
    output reg        tx_pin,
    output reg        busy
);

    localparam CLKS_PER_BIT = 16'd434; // 50MHz / 115200 baud
    
    reg [15:0] clk_count;
    reg [3:0]  bit_idx;
    reg [9:0]  shift_reg; 

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_pin    <= 1'b1; 
            busy      <= 1'b0;
            clk_count <= 16'd0;
            bit_idx   <= 4'd0;
        end else begin
            if (tx_req && !busy) begin
                shift_reg <= {1'b1, tx_data, 1'b0}; 
                busy      <= 1'b1;
                clk_count <= 16'd0;
                bit_idx   <= 4'd0;
            end else if (busy) begin
                if (clk_count < CLKS_PER_BIT - 16'd1) begin
                    clk_count <= clk_count + 16'd1;
                end else begin
                    clk_count <= 16'd0;
                    tx_pin    <= shift_reg[0];
                    shift_reg <= {1'b1, shift_reg[9:1]};
                    
                    if (bit_idx < 4'd9) begin
                        bit_idx <= bit_idx + 4'd1;
                    end else begin
                        busy   <= 1'b0;
                        tx_pin <= 1'b1;
                    end
                end
            end
        end
    end
endmodule
